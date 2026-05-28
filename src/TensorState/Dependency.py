"""Dependency graph and apoptotic pruning primitives.

This module combines two formerly separate implementations:

1. The graph-based dependency tracing from the `feat/dependency` branch
   (originally inspired by [torch-pruning](https://github.com/VainF/Torch-Pruning)).
2. The merge-then-destroy apoptosis primitives from `neurofilament/apoptosis.py`.

The result is a single canonical module with three layers:

- **Candidate generation** (functions): identify which groups of neurons
  are redundant. `zero_info_groups` finds always-off / always-on /
  perfectly-synchronized neurons from a binary state matrix.
  `correlated_weight_groups` filters those candidates by weight correlation.
- **Per-node merge + destroy** (methods on each ElementNode subclass):
  the layer-type-specific weight surgery. ``merge_outputs`` and
  ``merge_inputs`` combine sibling neurons' weights; ``destroy_outputs``
  and ``destroy_inputs`` remove the redundant rows / columns.
- **Graph-level orchestration** (`GroupGraph.apoptose`): walks the graph
  to apply merge + destroy across a chain of connected layers in the
  correct linearity-preserving order (mean for producing, sum for
  consuming).

Method naming convention (renamed from the original Dependency.py):

- ``destroy_outputs(idxs)`` — was ``apoptosis(idxs)``. Removes output rows.
- ``destroy_inputs(idxs)`` — was ``prune(idxs)``. Removes input columns.
- ``merge_outputs(groups)`` — NEW. Combines sibling output neurons.
- ``merge_inputs(groups)`` — NEW. Sums sibling input connections.

At the graph level:

- ``GroupGraph.destroy(idxs)`` — was ``apoptosis(idxs)``. Destroy across chain.
- ``GroupGraph.apoptose(groups)`` — NEW. Merge then destroy.
"""

from __future__ import annotations

from enum import IntFlag
from typing import Any, ClassVar

import numpy as np
import torch
import torchvision
from grandalf.graphs import Edge, Graph, Vertex, graph_core
from pydantic import BaseModel, ConfigDict

module_io = torch.Tensor | tuple[torch.Tensor, ...]


class NodeError(Exception):
    """Raised when a node cannot accept the given ModuleData/GradientData."""


class ApoptosisType(IntFlag):
    """Flags controlling which signals the apoptose pipeline uses.

    Mirrors the enum in the legacy ``neurofilament/apoptosis.py``. The
    bits compose: ``ApoptosisType.weights | ApoptosisType.connections``
    means apply both the weight-correlation filter on the producing
    layer and the connection-correlation filter on the consuming layer.
    """

    states = 0  # state-correlation only (no weight filter)
    weights = 1  # add weight correlation on producing layer
    connections = 2  # add weight correlation on consuming layer
    wc = weights | connections


# =============================================================================
# Data carriers
# =============================================================================


class GradientData(BaseModel):
    """Data attached to a node when only the grad_fn is known."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    grad_fn: torch.autograd.graph.Node


class ModuleData(GradientData):
    """Data attached to a node that maps back to a torch.nn.Module."""

    module: torch.nn.Module


# =============================================================================
# Candidate generation
# =============================================================================


def zero_info_groups(states: np.ndarray) -> list[list[int]]:
    """Find groups of neurons that contribute no entropy.

    Three categories are returned in a single flat list:

    1. **Always-off**: neurons with zero variance whose first observation
       is 0. These never fire.
    2. **Always-on**: neurons with zero variance whose first observation
       is 1. These always fire.
    3. **Perfectly-correlated clusters**: groups of neurons whose firing
       patterns are essentially identical (|correlation| >= 1 - 1/N^2).
       Each cluster has more than one neuron.

    Used by the apoptose pipeline as the first filter on candidate
    neurons to merge. Ported from
    ``neurofilament/apoptosis.py:zero_info_neurons``.

    Args:
        states: A 2D boolean array of shape ``(n_observations, n_neurons)``.

    Returns:
        A list of lists of neuron indices. The first two entries are the
        always-off and always-on groups (possibly empty). Remaining
        entries are correlated clusters with > 1 member each.
    """
    if states.ndim != 2:
        raise ValueError("states must be a 2D array")
    if states.shape[0] < 2:
        return [[], []]

    corr = np.corrcoef(states.T)

    # NaN correlation rows indicate zero-variance neurons (constants).
    # Use the first observation to decide off vs on, but verify against
    # all observations to avoid misclassifying a near-constant neuron
    # whose first sample happens to be an outlier.
    all_nan = np.isnan(corr).all(axis=0)
    off_neurons = np.argwhere(all_nan & ~states.any(axis=0)).flatten().tolist()
    on_neurons = np.argwhere(all_nan & states.all(axis=0)).flatten().tolist()

    # Correlated clusters: threshold just below 1 to absorb numerical noise.
    threshold = 1.0 - 1.0 / states.shape[0] ** 2
    adjacency = np.abs(corr) >= threshold
    np.fill_diagonal(adjacency, False)  # noqa: FBT003

    groups: list[list[int]] = [off_neurons, on_neurons]
    seen: set[int] = set(off_neurons) | set(on_neurons)
    for i in range(adjacency.shape[0]):
        if i in seen:
            continue
        cluster = [
            int(j) for j in np.argwhere(adjacency[i]).flatten() if int(j) not in seen
        ]
        if not cluster:
            continue
        cluster = [i, *cluster]
        for j in cluster:
            seen.add(j)
        groups.append(cluster)

    return groups


def correlated_weight_groups(
    weight: torch.Tensor,
    candidates: list[list[int]],
    threshold: float = 0.9,
    axis: str = "output",
) -> list[list[int]]:
    """Filter candidate groups by weight correlation.

    For each candidate group, compute pairwise weight correlation among
    members; keep sub-groups whose pairwise correlations exceed
    ``threshold``. A candidate group of size 1 is dropped.

    Ported and refined from
    ``neurofilament/apoptosis.py:correlated_neuron_weights``.

    Args:
        weight: The layer's weight tensor (any shape; will be flattened
            per-output-neuron).
        candidates: List of candidate index groups, typically produced
            by :func:`zero_info_groups`.
        threshold: Minimum pairwise correlation (default 0.9).
        axis: ``"output"`` correlates along axis 0 (producing-layer
            neurons), ``"input"`` correlates along axis 1 (consuming
            layer's input connections).

    Returns:
        A list of refined groups, each with ≥ 2 members and pairwise
        weight correlation ≥ ``threshold``.
    """
    if axis not in ("output", "input"):
        raise ValueError("axis must be 'output' or 'input'")

    weight_np = weight.detach().cpu().numpy()
    if axis == "input":
        # Move input dim to axis 0 so we can flatten "per-input-channel"
        weight_np = np.swapaxes(weight_np, 0, 1)
    # Flatten remaining dims so each row is one neuron's full weight vector.
    weight_np = weight_np.reshape(weight_np.shape[0], -1)

    refined: list[list[int]] = []
    for group in candidates:
        if len(group) < 2:
            continue
        rows = weight_np[group]
        corr = np.corrcoef(rows)
        # Positive correlation only: anti-correlated neurons encode
        # complementary (not redundant) information and should not be
        # merged. Matches the original neurofilament semantics.
        adjacency = corr > threshold
        np.fill_diagonal(adjacency, False)  # noqa: FBT003
        seen_local: set[int] = set()
        for i in range(adjacency.shape[0]):
            if i in seen_local:
                continue
            members = [
                int(j)
                for j in np.argwhere(adjacency[i]).flatten()
                if int(j) not in seen_local
            ]
            if not members:
                continue
            cluster = [group[i], *(group[j] for j in members)]
            for j in [i, *members]:
                seen_local.add(j)
            refined.append(cluster)

    return refined


# =============================================================================
# Graph node hierarchy
# =============================================================================


class ElementNode(Vertex):
    """Base node type. Specialized via the ``@ElementNode.register`` decorator.

    Subclasses override the merge / destroy methods to implement the
    layer-type-specific weight surgery. The default base implementations
    are no-ops so generic traversal nodes (Permute, AdaptivePool, etc.)
    can sit in the chain without doing anything.

    Method conventions:

    - ``destroy_outputs(idxs)``: remove the layer's output neurons at
      ``idxs``. Returns ``(keep_idxs, idxs)``.
    - ``destroy_inputs(idxs)``: remove the layer's input connections at
      ``idxs``. Returns ``(keep_idxs, idxs)``.
    - ``merge_outputs(groups)``: combine sibling output neurons. The
      first index of each group is the survivor; weights of the rest
      are averaged into it. Returns the list of redundant indices
      (suitable for passing to ``destroy_outputs``).
    - ``merge_inputs(groups)``: combine sibling input connections. The
      first index of each group is the survivor; weights of the rest
      are summed into it (linearity preservation). Returns the list of
      redundant indices.
    """

    _module_type: ClassVar[tuple[type[torch.nn.Module] | str, ...]] = ()
    _data_type: ClassVar[tuple[type, ...]] = (ModuleData, GradientData)
    traverse_node: bool = True
    data: GradientData | ModuleData
    TYPES: ClassVar[dict[type[torch.nn.Module] | str, OpNode]] = {}

    def __init__(self, data: GradientData | ModuleData):
        assert isinstance(data, self._data_type)
        if len(self._module_type) > 0:
            success = False
            for mtype in self._module_type:
                if isinstance(mtype, str):
                    if mtype in data.grad_fn.name().lower():
                        success = True
                        break
                elif isinstance(data, ModuleData) and isinstance(data.module, mtype):
                    success = True
                    break
            if not success:
                raise NodeError

        super().__init__(data)

    def __str__(self) -> str:
        return str(f"class={self.__class__.__name__}, {self.data}")

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def register(
        cls,
        module_type: type[torch.nn.Module]
        | str
        | tuple[type[torch.nn.Module] | str, ...],
        func: ElementNode | None = None,
    ):
        if func is None:
            return lambda x: cls.register(module_type=module_type, func=x)

        assert module_type not in cls.TYPES, f"Module {module_type} already registered."

        if not isinstance(module_type, tuple):
            module_type = (module_type,)

        func._module_type = ()
        for mt in module_type:
            cls.TYPES[mt] = func
            func._module_type = (*func._module_type, mt)
        return func

    @classmethod
    def _default_new(cls, data):
        return ElementNode(data)

    @classmethod
    def new(cls, data: GradientData | ModuleData):
        cls_types = tuple(t for t in cls.TYPES if not isinstance(t, str))
        str_types = tuple(t for t in cls.TYPES if isinstance(t, str))
        if isinstance(data, ModuleData) and isinstance(data.module, tuple(cls_types)):
            for t, c in cls.TYPES.items():
                if isinstance(t, str):
                    continue
                try:
                    if isinstance(data.module, t):
                        return c(data)
                except NodeError:
                    pass
        elif any(s in data.grad_fn.name().lower() for s in str_types):
            for s, c in cls.TYPES.items():
                if not isinstance(s, str):
                    continue
                try:
                    if s in data.grad_fn.name().lower():
                        return c(data)
                except NodeError:
                    pass

        return cls._default_new(data)

    def _upstream_dendrites(self, vertex: ElementNode | OpNode):
        for edge in vertex.e_in():
            vertex = edge.v[0]
            if vertex.traverse_node:
                dendrites = vertex._upstream_dendrites(vertex)
                if dendrites is not None:
                    return dendrites
            else:
                return vertex.neurons()

        return 0

    def _downstream_neurons(self, vertex: ElementNode | OpNode):
        for edge in vertex.e_out():
            vertex = edge.v[1]
            if vertex.traverse_node:
                neurons = vertex._downstream_neurons(vertex)
                if neurons is not None:
                    return neurons
            else:
                return vertex.dendrites()

        return 0

    def dendrites(self):
        try:
            return self._upstream_dendrites(self)
        except RecursionError:
            print(self)
            raise

    def neurons(self):
        try:
            return self._downstream_neurons(self)
        except RecursionError:
            print(self)
            raise

    # --- Default merge/destroy: no-op (for nodes that don't own params) ----

    def destroy_outputs(self, idxs: list[int]):
        """Remove output neurons at ``idxs``. Base impl is a no-op."""
        return None, idxs

    def destroy_inputs(self, idxs: list[int]):
        """Remove input connections at ``idxs``. Base impl is a no-op."""
        return None, idxs

    def merge_outputs(self, groups: list[list[int]]) -> list[int]:
        """Combine sibling output neurons into the first member of each group.

        Returns the list of redundant indices (the survivor of each group is
        NOT in this list). Base impl is a no-op.
        """
        return _redundant_idxs_from_groups(groups)

    def merge_inputs(self, groups: list[list[int]]) -> list[int]:
        """Combine sibling input connections into the first member of each group.

        Sums weight columns into the survivor. Returns the redundant indices.
        Base impl is a no-op.
        """
        return _redundant_idxs_from_groups(groups)

    def linked_neurons(
        self,
        idxs: tuple[tuple[int], ...] | tuple[tuple[tuple, ...], ...] | None = None,
    ) -> tuple[tuple[int], ...] | tuple[tuple[tuple, ...], ...]:
        if idxs is None:
            return tuple((i,) for i in range(self.neurons()))
        if len(idxs) != self.neurons():
            raise ValueError(
                f"Node: {self}\n"
                f"    Expected idxs to be size {self.neurons()} but was "
                f"size {len(idxs)}"
            )
        return idxs

    def and_index(self, idxs: list[int], neurons: int | None = None):
        if neurons is None:
            neurons = self.neurons()

        scale = self.dendrites() / neurons

        if scale == 1:
            return idxs
        if scale > 1:
            out_idxs = []
            for idx in idxs:
                out_idxs.extend([idx + scale * i for i in range(int(scale))])
        else:
            dendrites = self.dendrites()
            out_idxs = [idx for idx in idxs if idx < dendrites]
            assert len(out_idxs) / len(idxs) == scale, (
                f"{self.__class__.__name__}: scale={scale}, "
                f"dendrites={self.dendrites()}, neurons={neurons}"
            )
            for idx in out_idxs:
                for i in range(int(scale)):
                    assert (idx + scale * i) in idxs

        return out_idxs


class OpNode(ElementNode):
    """Op node — terminates graph traversal because it owns parameters."""

    _module_type: ClassVar[tuple[type[torch.nn.Module] | str, ...]] = (torch.nn.Module,)
    _data_type: ClassVar[tuple[type, ...]] = (ModuleData,)
    traverse_node: bool = False
    TYPES: ClassVar[dict[type[torch.nn.Module] | str, OpNode]] = {}

    @classmethod
    def _default_new(cls, data):
        return ElementNode.new(data)

    def dendrites(self):
        raise NotImplementedError(
            f"Not implemented for class {self.__class__.__name__}"
        )

    def neurons(self):
        raise NotImplementedError(
            f"Not implemented for class {self.__class__.__name__}"
        )

    def destroy_outputs(self, idxs: list[int]):
        raise NotImplementedError(
            f"Not implemented for class {self.__class__.__name__}"
        )

    def destroy_inputs(self, idxs: list[int]):
        raise NotImplementedError(
            f"Not implemented for class {self.__class__.__name__}"
        )

    def merge_outputs(self, groups: list[list[int]]) -> list[int]:
        raise NotImplementedError(
            f"Not implemented for class {self.__class__.__name__}"
        )

    def merge_inputs(self, groups: list[list[int]]) -> list[int]:
        raise NotImplementedError(
            f"Not implemented for class {self.__class__.__name__}"
        )

    def and_index(self, idxs: list[int], neurons: int | None = None):
        return None


# =============================================================================
# Helpers
# =============================================================================


def _redundant_idxs_from_groups(groups: list[list[int]]) -> list[int]:
    """Given groups of indices, return everything except the first of each."""
    out: list[int] = []
    for group in groups:
        if len(group) > 1:
            out.extend(group[1:])
    return sorted(set(out))


# =============================================================================
# Specialized nodes — BatchNorm
# =============================================================================


@ElementNode.register(torch.nn.modules.batchnorm._BatchNorm)
class BatchNormNode(ElementNode):
    def dendrites(self):
        return self.data.module.num_features

    neurons = dendrites

    def merge_outputs(self, groups: list[list[int]]) -> list[int]:
        """Average weight/bias/running stats across each group.

        Stores the merged value into the first member; other members are
        returned as redundant indices.
        """
        layer = self.data.module
        for group in groups:
            if len(group) < 2:
                continue
            dead = group[0]
            tensors = []
            for attr in ("weight", "bias"):
                t = getattr(layer, attr, None)
                if t is not None:
                    tensors.append(t)
            for attr in ("running_mean", "running_var"):
                t = getattr(layer, attr, None)
                if t is not None:
                    tensors.append(t)
            for t in tensors:
                t.data[dead] = t.data[group].mean(dim=0)
        return _redundant_idxs_from_groups(groups)

    merge_inputs = merge_outputs  # BatchNorm has no separate input dim

    def destroy_outputs(self, idxs: list[int]):
        layer = self.data.module
        keep_idxs = sorted(set(range(layer.num_features)) - set(idxs))
        layer.num_features = layer.num_features - len(idxs)

        if layer.running_mean is not None:
            layer.running_mean = layer.running_mean.data.clone()[keep_idxs]
        if layer.running_var is not None:
            layer.running_var = layer.running_var.data.clone()[keep_idxs]
        if layer.affine:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
            layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])

        return keep_idxs, idxs

    destroy_inputs = destroy_outputs


# =============================================================================
# Specialized nodes — GroupNorm / LayerNorm
# =============================================================================


@ElementNode.register(torch.nn.GroupNorm)
class GroupNormNode(ElementNode):
    """GroupNorm affine params are per-channel; merge / destroy along channels."""

    def dendrites(self):
        return self.data.module.num_channels

    neurons = dendrites

    def merge_outputs(self, groups: list[list[int]]) -> list[int]:
        layer = self.data.module
        if not layer.affine:
            return _redundant_idxs_from_groups(groups)
        for group in groups:
            if len(group) < 2:
                continue
            dead = group[0]
            layer.weight.data[dead] = layer.weight.data[group].mean(dim=0)
            layer.bias.data[dead] = layer.bias.data[group].mean(dim=0)
        return _redundant_idxs_from_groups(groups)

    merge_inputs = merge_outputs

    def destroy_outputs(self, idxs: list[int]):
        layer = self.data.module
        keep_idxs = sorted(set(range(layer.num_channels)) - set(idxs))
        layer.num_channels = layer.num_channels - len(idxs)
        # num_groups must still divide num_channels evenly. If not, downstream
        # behaviour is undefined; warn the caller via assertion.
        assert layer.num_channels % layer.num_groups == 0, (
            "GroupNorm num_channels must remain divisible by num_groups after destroy"
        )

        if layer.affine:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
            layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
        return keep_idxs, idxs

    destroy_inputs = destroy_outputs


@ElementNode.register(torch.nn.LayerNorm)
class LayerNormNode(ElementNode):
    """LayerNorm affine params are along the last normalized dimension."""

    def dendrites(self):
        # LayerNorm normalizes over normalized_shape (possibly multi-dim).
        # We support the common case where normalized_shape is a single int
        # (last-dim normalization).
        shape = self.data.module.normalized_shape
        if isinstance(shape, int):
            return shape
        if len(shape) == 1:
            return shape[0]
        raise NotImplementedError(
            "LayerNormNode merge/destroy not implemented for multi-dim "
            f"normalized_shape; got {shape}"
        )

    neurons = dendrites

    def merge_outputs(self, groups: list[list[int]]) -> list[int]:
        layer = self.data.module
        if not layer.elementwise_affine:
            return _redundant_idxs_from_groups(groups)
        for group in groups:
            if len(group) < 2:
                continue
            dead = group[0]
            layer.weight.data[dead] = layer.weight.data[group].mean(dim=0)
            if layer.bias is not None:
                layer.bias.data[dead] = layer.bias.data[group].mean(dim=0)
        return _redundant_idxs_from_groups(groups)

    merge_inputs = merge_outputs

    def destroy_outputs(self, idxs: list[int]):
        layer = self.data.module
        n = self.dendrites()
        keep_idxs = sorted(set(range(n)) - set(idxs))
        new_n = n - len(idxs)
        layer.normalized_shape = (
            (new_n,) if isinstance(layer.normalized_shape, tuple) else new_n
        )

        if layer.elementwise_affine:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
            if layer.bias is not None:
                layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
        return keep_idxs, idxs

    destroy_inputs = destroy_outputs


# =============================================================================
# Specialized nodes — pooling / reshape / permute (no params)
# =============================================================================


@ElementNode.register(
    (
        torch.nn.modules.pooling._AdaptiveMaxPoolNd,
        torch.nn.modules.pooling._AdaptiveAvgPoolNd,
    )
)
class AdaptivePoolNode(ElementNode):
    pass


@ElementNode.register(torchvision.ops.misc.Permute)
class PermuteNode(ElementNode):
    pass


@ElementNode.register(("reshape", "view"))
class ReshapeNode(ElementNode):
    traverse_node = False

    def linked_neurons(
        self,
        idxs: tuple[tuple[int], ...] | tuple[tuple[tuple, ...], ...] | None = None,
    ):
        idxs = super().linked_neurons(idxs)
        dendrites = self.dendrites()
        return tuple(tuple(idxs[i::dendrites]) for i in range(dendrites))


# =============================================================================
# Specialized nodes — Conv (regular + depthwise group)
# =============================================================================


@OpNode.register(torch.nn.modules.conv._ConvNd)
class ConvNode(OpNode):
    def __init__(self, data: GradientData | ModuleData):
        super().__init__(data)
        assert isinstance(data, ModuleData)
        if data.module.groups != 1:
            raise NodeError

    def dendrites(self):
        return self.data.module.in_channels

    def neurons(self):
        return self.data.module.out_channels

    def merge_outputs(self, groups: list[list[int]]) -> list[int]:
        """Average weights and biases across each group's output channels.

        Linearity rule: the producing-layer weights are *averaged* so the
        merged neuron's output equals the average of the originals.
        """
        layer = self.data.module
        for group in groups:
            if len(group) < 2:
                continue
            dead = group[0]
            if not layer.transposed:
                layer.weight.data[dead] = layer.weight.data[group].mean(dim=0)
            else:
                layer.weight.data[:, dead] = layer.weight.data[:, group].mean(dim=1)
            if layer.bias is not None:
                layer.bias.data[dead] = layer.bias.data[group].mean(dim=0)
        return _redundant_idxs_from_groups(groups)

    def merge_inputs(self, groups: list[list[int]]) -> list[int]:
        """Sum input weight columns across each group.

        Linearity rule: the consuming-layer weights are *summed* so the new
        effective weight is `sum_i W_i` instead of `W_i * mean_signal`.
        """
        layer = self.data.module
        for group in groups:
            if len(group) < 2:
                continue
            dead = group[0]
            if not layer.transposed:
                layer.weight.data[:, dead] = layer.weight.data[:, group].sum(dim=1)
            else:
                layer.weight.data[dead] = layer.weight.data[group].sum(dim=0)
        return _redundant_idxs_from_groups(groups)

    def destroy_outputs(self, idxs: list[int]):
        layer = self.data.module
        keep_idxs = sorted(set(range(layer.out_channels)) - set(idxs))
        layer.out_channels -= len(idxs)

        if not layer.transposed:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
        else:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[:, keep_idxs])
        if layer.bias is not None:
            layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])

        return keep_idxs, idxs

    def destroy_inputs(self, idxs: list[int]):
        layer = self.data.module
        keep_idxs = sorted(set(range(layer.in_channels)) - set(idxs))
        layer.in_channels = layer.in_channels - len(idxs)

        if not layer.transposed:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[:, keep_idxs])
        else:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])

        return keep_idxs, idxs


@ElementNode.register(torch.nn.modules.conv._ConvNd)
class ConvGroupNode(ElementNode):
    """Depthwise / fully-grouped conv: in_channels == groups."""

    def __init__(self, data: GradientData | ModuleData):
        super().__init__(data)
        assert isinstance(data, ModuleData)
        if (
            data.module.groups != data.module.in_channels
            and data.module.in_channels != data.module.out_channels
        ):
            raise NodeError

    def dendrites(self):
        return self.data.module.in_channels

    def neurons(self):
        return self.data.module.out_channels

    def merge_outputs(self, groups: list[list[int]]) -> list[int]:
        layer = self.data.module
        for group in groups:
            if len(group) < 2:
                continue
            dead = group[0]
            if not layer.transposed:
                layer.weight.data[dead] = layer.weight.data[group].mean(dim=0)
            else:
                layer.weight.data[:, dead] = layer.weight.data[:, group].mean(dim=1)
            if layer.bias is not None:
                layer.bias.data[dead] = layer.bias.data[group].mean(dim=0)
        return _redundant_idxs_from_groups(groups)

    merge_inputs = merge_outputs  # depthwise: in_channel == out_channel per group

    def destroy_outputs(self, idxs: list[int]):
        layer = self.data.module
        keep_idxs = sorted(set(range(layer.out_channels)) - set(idxs))
        layer.out_channels -= len(idxs)

        if not layer.transposed:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
        else:
            layer.weight = torch.nn.Parameter(layer.weight.data.clone()[:, keep_idxs])
        if layer.bias is not None:
            layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])

        layer.in_channels = layer.out_channels
        layer.groups = layer.out_channels

        return keep_idxs, idxs

    destroy_inputs = destroy_outputs


# =============================================================================
# Specialized nodes — Linear
# =============================================================================


@OpNode.register(torch.nn.Linear)
class LinearNode(OpNode):
    def dendrites(self):
        return self.data.module.in_features

    def neurons(self):
        return self.data.module.out_features

    def merge_outputs(self, groups: list[list[int]]) -> list[int]:
        """Average weight rows + bias entries across each group."""
        layer = self.data.module
        for group in groups:
            if len(group) < 2:
                continue
            dead = group[0]
            layer.weight.data[dead] = layer.weight.data[group].mean(dim=0)
            if layer.bias is not None:
                layer.bias.data[dead] = layer.bias.data[group].mean(dim=0)
        return _redundant_idxs_from_groups(groups)

    def merge_inputs(self, groups: list[list[int]]) -> list[int]:
        """Sum weight columns across each group's input features."""
        layer = self.data.module
        for group in groups:
            if len(group) < 2:
                continue
            dead = group[0]
            layer.weight.data[:, dead] = layer.weight.data[:, group].sum(dim=1)
        return _redundant_idxs_from_groups(groups)

    def destroy_outputs(self, idxs: list[int]):
        layer = self.data.module
        keep_idxs = sorted(set(range(layer.out_features)) - set(idxs))
        layer.out_features -= len(idxs)
        layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
        if layer.bias is not None:
            layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
        return keep_idxs, idxs

    def destroy_inputs(self, idxs: list[int]):
        layer = self.data.module
        keep_idxs = sorted(set(range(layer.in_features)) - set(idxs))
        layer.in_features = layer.in_features - len(idxs)
        layer.weight = torch.nn.Parameter(layer.weight.data.clone()[:, keep_idxs])
        return keep_idxs, idxs


# =============================================================================
# Specialized nodes — MultiheadAttention (head-level, EXACT-ONLY)
# =============================================================================


@OpNode.register(torch.nn.MultiheadAttention)
class AttentionNode(OpNode):
    """Head-level merge / destroy for ``nn.MultiheadAttention``.

    The merge unit is a **head**, not a channel. ``in_proj_weight`` is the
    packed projection tensor of shape ``(3*E, E)`` = ``[W_q; W_k; W_v]``
    stacked as three row-blocks; ``in_proj_bias`` is ``(3*E,)``. With
    ``hd = E // num_heads`` head ``h`` owns rows ``h*hd:(h+1)*hd`` within
    *each* of the three blocks (block offsets ``0``, ``E``, ``2E``) plus
    ``out_proj.weight`` *columns* ``h*hd:(h+1)*hd``.

    EXACT-ONLY scope (ratified decision — do not exceed):
    ``softmax(QKᵀ / √d)`` is non-linear in the projection weights, so the
    forward output is preserved exactly only when

    - **destroying** dead / duplicate heads (their Q/K/V projections and
      ``out_proj`` columns are removed), or
    - **merging IDENTICAL heads** — heads whose Q/K/V projection blocks are
      equal. Averaging the producer (Q/K/V) blocks of identical heads is a
      no-op on each head's output; summing the consumer (``out_proj``)
      column-blocks then reproduces the original ``(W_a + W_b) @ o``
      contribution.

    Approximate merging of *distinct* heads is intentionally NOT
    implemented: there is no linear weight surgery that preserves the
    softmax mixture of two different heads.
    """

    def _hd(self) -> int:
        layer = self.data.module
        return layer.embed_dim // layer.num_heads

    def dendrites(self):
        return self.data.module.embed_dim

    neurons = dendrites

    def _head_rows(self, head: int, hd: int, embed_dim: int) -> list[int]:
        """Packed-tensor row indices owned by ``head`` across Q/K/V blocks."""
        rows: list[int] = []
        for block in range(3):
            off = block * embed_dim
            rows.extend(range(off + head * hd, off + (head + 1) * hd))
        return rows

    def merge_outputs(self, groups: list[list[int]]) -> list[int]:
        """Merge IDENTICAL heads: mean producer (Q/K/V), sum consumer (out_proj).

        Each group lists head indices; the first is the survivor. The
        survivor's Q/K/V row-blocks + bias slices become the mean of the
        group's heads and its ``out_proj`` column-block becomes their sum.
        Only exact (identical-head) merges preserve the forward output.
        """
        layer = self.data.module
        e = layer.embed_dim
        hd = self._hd()
        w = layer.in_proj_weight.data
        b = layer.in_proj_bias.data if layer.in_proj_bias is not None else None
        out_w = layer.out_proj.weight.data
        for group in groups:
            if len(group) < 2:
                continue
            survivor = group[0]
            for block in range(3):
                off = block * e
                surv_sl = slice(off + survivor * hd, off + (survivor + 1) * hd)
                stacked = torch.stack(
                    [w[off + g * hd : off + (g + 1) * hd] for g in group]
                )
                w[surv_sl] = stacked.mean(dim=0)
                if b is not None:
                    stacked_b = torch.stack(
                        [b[off + g * hd : off + (g + 1) * hd] for g in group]
                    )
                    b[surv_sl] = stacked_b.mean(dim=0)
            surv_cols = slice(survivor * hd, (survivor + 1) * hd)
            out_w[:, surv_cols] = torch.stack(
                [out_w[:, g * hd : (g + 1) * hd] for g in group], dim=0
            ).sum(dim=0)
        return _redundant_idxs_from_groups(groups)

    def merge_inputs(self, groups: list[list[int]]) -> list[int]:
        """No-op merge for inputs.

        Head merging acts on the attention's own packed projection /
        out_proj weights via :meth:`merge_outputs`; there is no separate
        input-side weight to combine.
        """
        return _redundant_idxs_from_groups(groups)

    def destroy_outputs(self, idxs: list[int]):
        """Remove heads ``idxs``: drop their Q/K/V rows, bias slices, out cols.

        Decrements ``num_heads`` and ``embed_dim`` and asserts
        ``embed_dim % num_heads == 0`` afterwards (analogous to GroupNorm).
        """
        layer = self.data.module
        e = layer.embed_dim
        hd = self._hd()
        num_heads = layer.num_heads

        keep_heads = sorted(set(range(num_heads)) - set(idxs))
        new_num_heads = num_heads - len(idxs)
        new_e = e - len(idxs) * hd

        # Rows to keep within the packed (3*E, E) tensor, block by block.
        keep_rows: list[int] = []
        for block in range(3):
            off = block * e
            for head in keep_heads:
                keep_rows.extend(range(off + head * hd, off + (head + 1) * hd))
        # Columns kept on the embed (input) dimension.
        keep_cols: list[int] = []
        for head in keep_heads:
            keep_cols.extend(range(head * hd, (head + 1) * hd))

        new_in_proj_w = layer.in_proj_weight.data.clone()[keep_rows][:, keep_cols]
        layer.in_proj_weight = torch.nn.Parameter(new_in_proj_w)
        if layer.in_proj_bias is not None:
            layer.in_proj_bias = torch.nn.Parameter(
                layer.in_proj_bias.data.clone()[keep_rows]
            )

        # out_proj is Linear(E, E): drop both the column-blocks (inputs, the
        # per-head concat) and the rows (its outputs are the embed dim).
        out_proj = layer.out_proj
        new_out_w = out_proj.weight.data.clone()[keep_cols][:, keep_cols]
        out_proj.weight = torch.nn.Parameter(new_out_w)
        out_proj.in_features = new_e
        out_proj.out_features = new_e
        if out_proj.bias is not None:
            out_proj.bias = torch.nn.Parameter(out_proj.bias.data.clone()[keep_cols])

        layer.num_heads = new_num_heads
        layer.embed_dim = new_e
        layer.kdim = new_e
        layer.vdim = new_e
        layer.head_dim = hd

        assert layer.embed_dim % layer.num_heads == 0, (
            "MultiheadAttention embed_dim must remain divisible by num_heads "
            "after destroy"
        )
        return keep_heads, idxs

    destroy_inputs = destroy_outputs


# =============================================================================
# Edges and graph types
# =============================================================================


class Dependency(Edge):
    x: OpNode | ElementNode
    y: OpNode | ElementNode
    w: int | float = 1
    data: Any | None = None
    connect: bool = False

    def __init__(
        self,
        x: OpNode | ElementNode,
        y: OpNode | ElementNode,
        w: int | float = 1,
        data: Any | None = None,
        connect: bool = False,  # noqa: FBT001, FBT002 -- grandalf Edge API
    ):
        assert isinstance(x, (OpNode, ElementNode))
        assert isinstance(y, (OpNode, ElementNode))
        super().__init__(x, y, w, data, connect)

    def __str__(self) -> str:
        return str(self.data)

    def __repr__(self) -> str:
        return str(self.data)


class GroupGraph(graph_core):
    """A connected component of the module graph.

    Provides three entry points:

    - :py:meth:`linked_neurons`: discover which neurons across the chain
      move together (e.g., a residual stretch's output channels).
    - :py:meth:`destroy`: remove specified neurons across the chain
      (chain-wide destroy without merging).
    - :py:meth:`apoptose`: combined merge-then-destroy for groups of
      linked neurons. The standard apoptosis entry point.
    """

    def destroy(self, idxs: list[int]):
        """Destroy connected neurons across the graph at ``idxs`` (no merge)."""
        stack_type = tuple[OpNode | ElementNode, list[int]]

        leaves = self.leaves()
        roots = self.roots()
        process_stack: list[stack_type] = [(leaf, idxs) for leaf in leaves]
        visited: set[OpNode | ElementNode] = set()

        destroy_stack: list[stack_type] = []

        while len(process_stack) > 0:
            node, indices = process_stack.pop()
            if node in visited:
                continue
            visited.add(node)

            new_indices = node.and_index(indices)
            for edge in node.e_in():
                vertex: OpNode | ElementNode = edge.v[0]
                process_stack.append(
                    (vertex, new_indices if new_indices is not None else indices)
                )

            if isinstance(node.data, ModuleData):
                destroy_stack.append((node, indices))

        visited = set()
        for node, indices in destroy_stack:
            if node in visited:
                continue
            visited.add(node)
            # Single-dim nodes (BatchNorm/GroupNorm/LayerNorm/Pool/Permute)
            # alias `destroy_inputs = destroy_outputs` at class level because
            # outputs and inputs are the same channel dim. Detect the alias
            # so we don't double-apply the surgery on interior nodes.
            single_dim = node.destroy_outputs.__func__ is node.destroy_inputs.__func__
            called_outputs = False
            if node not in leaves:
                node.destroy_outputs(indices)
                called_outputs = True
            if node not in roots and not (single_dim and called_outputs):
                node.destroy_inputs(indices)

    # Back-compat alias.
    apoptosis = destroy

    def apoptose(self, groups: list[list[int]]):
        """Merge + destroy along the chain for each group of linked output neurons.

        For each producing node in the chain, calls ``merge_outputs(groups)``
        to combine sibling weights into the first member of each group,
        then ``destroy_outputs(idxs)`` on the redundant indices. For each
        consuming node downstream, calls ``merge_inputs(groups)`` and
        ``destroy_inputs(idxs)``.

        Args:
            groups: A list of groups of linked output-neuron indices to
                merge. Each group's first index survives; the rest are
                destroyed.
        """
        if not groups:
            return

        idxs = _redundant_idxs_from_groups(groups)
        if not idxs:
            return

        stack_type = tuple[OpNode | ElementNode, list[list[int]]]
        leaves = self.leaves()
        roots = self.roots()
        process_stack: list[stack_type] = [(leaf, groups) for leaf in leaves]
        visited: set[OpNode | ElementNode] = set()
        apoptose_stack: list[stack_type] = []

        while len(process_stack) > 0:
            node, node_groups = process_stack.pop()
            if node in visited:
                continue
            visited.add(node)

            # Translate indices through reshape/pool nodes if they alter the
            # output-to-input mapping.
            flat = _redundant_idxs_from_groups(node_groups)
            new_flat = node.and_index(flat)
            if new_flat is not None and new_flat != flat:
                # Heuristic: if and_index reorders/scales, we conservatively
                # keep the original group structure for the upstream walk.
                # (Most layer types don't alter neuron identity.)
                pass

            for edge in node.e_in():
                vertex: OpNode | ElementNode = edge.v[0]
                process_stack.append((vertex, node_groups))

            if isinstance(node.data, ModuleData):
                apoptose_stack.append((node, node_groups))

        visited = set()
        for node, node_groups in apoptose_stack:
            if node in visited:
                continue
            visited.add(node)

            # Aliased nodes (BatchNorm/GroupNorm/LayerNorm/AttentionNode) set
            # `destroy_inputs = destroy_outputs` (and merge likewise) because
            # their input and output dims are the same neurons. On an interior
            # node both the output and input branches fire, so guard against
            # double-applying the surgery — mirrors `destroy()`.
            single_dim = node.destroy_outputs.__func__ is node.destroy_inputs.__func__
            redundant = _redundant_idxs_from_groups(node_groups)
            called_outputs = False
            if node not in leaves:
                node.merge_outputs(node_groups)
                node.destroy_outputs(redundant)
                called_outputs = True
            if node not in roots and not (single_dim and called_outputs):
                node.merge_inputs(node_groups)
                node.destroy_inputs(redundant)

    def linked_neurons(self):
        leaves = self.leaves()
        roots = self.roots()

        if len(list(self.V())) == 1 or leaves[0].dendrites() == roots[0].neurons():
            return [tuple([(i,) for i in range(leaves[0].neurons())])]

        if len(leaves) != 1:
            raise NotImplementedError("Multiple output nodes are not yet supported")

        process_stack = []
        for node in leaves:
            indices = [(i,) for i in range(node.dendrites())]
            for edge in node.e_in():
                process_stack.append((edge.v[0], indices))

        visited = set()
        linkages = []
        while len(process_stack) > 0:
            node, indices = process_stack.pop()
            if node in visited:
                continue
            visited.add(node)

            new_indices = node.linked_neurons(indices)
            for edge in node.e_in():
                vertex = edge.v[0]
                if vertex in roots:
                    linkages.append(new_indices)
                else:
                    process_stack.append((vertex, new_indices))

        return linkages


class ModuleGraph(Graph):
    model: torch.nn.Module
    _visit_count: ClassVar[dict[torch.nn.Module, int]] = {}
    _grad_trace: ClassVar[dict[Any, torch.nn.Module]] = {}
    _block_hook: bool = True

    def __init__(
        self,
        model: torch.nn.Module,
        inp_tensor: module_io | None = None,
    ):
        if inp_tensor is None:
            inp_tensor = torch.ones((1, 3, 256, 256))
        self.model = model
        self._visit_count = dict.fromkeys(model.modules(), 0)
        # Per-instance grad_fn -> module map. Reset here (not left as the shared
        # class-level dict) so graphs don't accumulate stale entries across
        # instances/tests — that cross-graph pollution made AlexNet apoptosis
        # tracing flaky under a full test run.
        self._grad_trace = {}

        # nn.MultiheadAttention is not a leaf (it owns an ``out_proj`` Linear
        # child) and its QKV projection is functional (``in_proj_weight`` is a
        # bare Parameter, not a submodule), so it would never surface as a
        # single graph node. Hook the whole MHA explicitly and SKIP its
        # ``out_proj`` child so the entire module maps to exactly one
        # ModuleData node instead of leaking an interior Linear node.
        mha_modules = [
            module
            for module in model.modules()
            if isinstance(module, torch.nn.MultiheadAttention)
        ]
        skip_modules = {id(m.out_proj) for m in mha_modules}

        hooks = [
            module.register_forward_hook(self)
            for module in model.modules()
            if (
                not list(module.children())
                or isinstance(module, torch.nn.MultiheadAttention)
            )
            and id(module) not in skip_modules
        ]

        self._block_hook = False
        self.model.eval()
        device = next(model.parameters()).device
        out: torch.Tensor = self.model(inp_tensor.to(device))
        self._block_hook = True

        for hook in hooks:
            hook.remove()

        # Trace the network and generate the nodes and edges
        gradients: list[torch.autograd.graph.Node] = [out.grad_fn]
        visited_nodes: list[torch.autograd.graph.Node] = []

        if out.grad_fn in self._grad_trace:
            node = OpNode.new(
                data=ModuleData(
                    module=self._grad_trace[out.grad_fn],
                    grad_fn=out.grad_fn,
                    name=out.grad_fn.name(),
                )
            )
        else:
            node = OpNode.new(
                data=GradientData(grad_fn=out.grad_fn, name=out.grad_fn.name())
            )

        nodes: dict[Any, OpNode | ElementNode] = {out.grad_fn: node}
        edges: list[Dependency] = []

        while len(gradients) > 0:
            grad_fn = gradients.pop()
            if grad_fn in visited_nodes:
                continue
            visited_nodes.append(grad_fn)

            node = nodes[grad_fn]

            for gf in grad_fn.next_functions:
                gf = gf[0]
                if gf is None:
                    continue
                if gf in nodes:
                    upstream_node = nodes[gf]
                else:
                    if gf in self._grad_trace:
                        upstream_node = OpNode.new(
                            data=ModuleData(
                                module=self._grad_trace[gf],
                                grad_fn=gf,
                                name=gf.name(),
                            )
                        )
                    else:
                        if "accumulategrad" in gf.name().lower():
                            continue
                        upstream_node = OpNode.new(
                            data=GradientData(grad_fn=gf, name=gf.name())
                        )

                    nodes[gf] = upstream_node

                edges.append(Dependency(x=upstream_node, y=node))
                gradients.append(gf)

        super().__init__(V=list(nodes.values()), E=edges)

        groups = Graph(V=list(nodes.values()), E=edges)
        vertices = list(groups.V())

        for vertex in vertices:
            if isinstance(vertex, OpNode):
                temp_node = OpNode.new(data=vertex.data)
                groups.add_vertex(temp_node)
                for edge in vertex.e_out():
                    temp_edge = Dependency(x=temp_node, y=edge.v[1])
                    groups.add_edge(temp_edge)
                    groups.remove_edge(edge)

        self.groups = []
        for component in groups.C:
            modules = [
                m.data.grad_fn for m in component.V() if isinstance(m.data, ModuleData)
            ]
            duplicates = [m for m in modules if modules.count(m) > 1]
            while len(duplicates) > 0:
                grad_fn = duplicates.pop()

                vertices: list[Vertex] = [
                    v
                    for v in component.V()
                    if isinstance(v.data, ModuleData) and v.data.grad_fn == grad_fn
                ]
                if len(vertices) == 0:
                    continue
                vertex = vertices.pop()
                dependencies = []
                for v in vertices:
                    for e in component.E():
                        if e.v[1] == v:
                            dependencies.append(Dependency(x=e.v[0], y=vertex))
                        elif e.v[0] == v:
                            dependencies.append(Dependency(x=vertex, y=e.v[1]))

                    for dep in dependencies:
                        component.add_edge(dep)

                    component.remove_vertex(v)

                duplicates = [d for d in duplicates if d != grad_fn]

            self.groups.append(GroupGraph(list(component.V()), list(component.E())))

    def __call__(
        self, module: torch.nn.Module, inputs: module_io, outputs: module_io
    ) -> None:
        if self._block_hook:
            raise RuntimeError(
                "Module is intended to be used as a forward hook and "
                "should not be directly called."
            )
        self._visit_count[module] += 1
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        grad_fn = outputs.grad_fn
        # An eval-mode ``nn.Dropout`` (and other pass-through ops) is the
        # identity: it returns the *same* tensor, so its output grad_fn is the
        # exact grad_fn object produced by the preceding module. A Dropout that
        # immediately follows a MultiheadAttention block would otherwise clobber
        # the MHA's entry here and steal its node. MHA owns its grad_fn — never
        # let a later pass-through alias overwrite it.
        if isinstance(self._grad_trace.get(grad_fn), torch.nn.MultiheadAttention):
            return
        self._grad_trace[grad_fn] = module
