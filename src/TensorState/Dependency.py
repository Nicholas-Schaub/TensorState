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
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np
import torch
import torchvision
from grandalf.graphs import Edge, Graph, Vertex, graph_core
from pydantic import BaseModel, ConfigDict

module_io = Union[torch.Tensor, Tuple[torch.Tensor, ...]]


class NodeError(Exception):
    """Raised when a node cannot accept the given ModuleData/GradientData."""


class ApoptosisType(IntFlag):
    """Flags controlling which signals the apoptose pipeline uses.

    Mirrors the enum in the legacy ``neurofilament/apoptosis.py``. The
    bits compose: ``ApoptosisType.weights | ApoptosisType.connections``
    means apply both the weight-correlation filter on the producing
    layer and the connection-correlation filter on the consuming layer.
    """

    states = 0          # state-correlation only (no weight filter)
    weights = 1         # add weight correlation on producing layer
    connections = 2     # add weight correlation on consuming layer
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


def zero_info_groups(states: np.ndarray) -> List[List[int]]:
    """Find groups of neurons that contribute no entropy.

    Three categories are returned in a single flat list:

    1. **Always-off**: neurons with zero variance whose first observation
       is 0. These never fire.
    2. **Always-on**: neurons with zero variance whose first observation
       is 1. These always fire.
    3. **Perfectly-correlated clusters**: groups of neurons whose firing
       patterns are essentially identical (|correlation| ≥ 1 − 1/N²).
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
    # all observations to avoid mis-classifying a near-constant neuron
    # whose first sample happens to be an outlier.
    all_nan = np.isnan(corr).all(axis=0)
    off_neurons = np.argwhere(all_nan & ~states.any(axis=0)).flatten().tolist()
    on_neurons = np.argwhere(all_nan & states.all(axis=0)).flatten().tolist()

    # Correlated clusters: threshold just below 1 to absorb numerical noise.
    threshold = 1.0 - 1.0 / states.shape[0] ** 2
    adjacency = np.abs(corr) >= threshold
    np.fill_diagonal(adjacency, False)

    groups: List[List[int]] = [off_neurons, on_neurons]
    seen: Set[int] = set(off_neurons) | set(on_neurons)
    for i in range(adjacency.shape[0]):
        if i in seen:
            continue
        cluster = [int(j) for j in np.argwhere(adjacency[i]).flatten() if int(j) not in seen]
        if not cluster:
            continue
        cluster = [i] + cluster
        for j in cluster:
            seen.add(j)
        groups.append(cluster)

    return groups


def correlated_weight_groups(
    weight: torch.Tensor,
    candidates: List[List[int]],
    threshold: float = 0.9,
    axis: str = "output",
) -> List[List[int]]:
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

    refined: List[List[int]] = []
    for group in candidates:
        if len(group) < 2:
            continue
        rows = weight_np[group]
        corr = np.corrcoef(rows)
        # Positive correlation only: anti-correlated neurons encode
        # complementary (not redundant) information and should not be
        # merged. Matches the original neurofilament semantics.
        adjacency = corr > threshold
        np.fill_diagonal(adjacency, False)
        seen_local: Set[int] = set()
        for i in range(adjacency.shape[0]):
            if i in seen_local:
                continue
            members = [int(j) for j in np.argwhere(adjacency[i]).flatten() if int(j) not in seen_local]
            if not members:
                continue
            cluster = [group[i]] + [group[j] for j in members]
            for j in [i] + members:
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

    _module_type: Tuple[Union[Type[torch.nn.Module], str], ...] = tuple()
    _data_type: Tuple[Type, ...] = (ModuleData, GradientData)
    traverse_node: bool = True
    data: Union[GradientData, ModuleData]
    TYPES: Dict[Union[Type[torch.nn.Module], str], OpNode] = {}

    def __init__(self, data: Union[GradientData, ModuleData]):
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
        module_type: Union[
            Type[torch.nn.Module], str, Tuple[Union[Type[torch.nn.Module], str], ...]
        ],
        func: Optional[ElementNode] = None,
    ):
        if func is None:
            return lambda x: cls.register(module_type=module_type, func=x)

        assert module_type not in cls.TYPES, f"Module {module_type} already registered."

        if not isinstance(module_type, tuple):
            module_type = (module_type,)

        func._module_type = tuple()
        for mt in module_type:
            cls.TYPES[mt] = func
            func._module_type = func._module_type + (mt,)
        return func

    @classmethod
    def _default_new(cls, data):
        return ElementNode(data)

    @classmethod
    def new(cls, data: Union[GradientData, ModuleData]):
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

    def _upstream_dendrites(self, vertex: Union[ElementNode, OpNode]):
        for edge in vertex.e_in():
            vertex = edge.v[0]
            if vertex.traverse_node:
                dendrites = vertex._upstream_dendrites(vertex)
                if dendrites is not None:
                    return dendrites
            else:
                return vertex.neurons()

        return 0

    def _downstream_neurons(self, vertex: Union[ElementNode, OpNode]):
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

    def destroy_outputs(self, idxs: List[int]):
        """Remove output neurons at ``idxs``. Base impl is a no-op."""
        return None, idxs

    def destroy_inputs(self, idxs: List[int]):
        """Remove input connections at ``idxs``. Base impl is a no-op."""
        return None, idxs

    def merge_outputs(self, groups: List[List[int]]) -> List[int]:
        """Combine sibling output neurons into the first member of each
        group. Returns the list of redundant indices (the survivor of
        each group is NOT in this list). Base impl is a no-op."""
        return _redundant_idxs_from_groups(groups)

    def merge_inputs(self, groups: List[List[int]]) -> List[int]:
        """Combine sibling input connections by summing weight columns
        into the first member of each group. Returns the redundant
        indices. Base impl is a no-op."""
        return _redundant_idxs_from_groups(groups)

    def linked_neurons(
        self,
        idxs: Union[Tuple[Tuple[int], ...], Tuple[Tuple[Tuple, ...], ...], None] = None,
    ) -> Union[Tuple[Tuple[int], ...], Tuple[Tuple[Tuple, ...], ...]]:
        if idxs is None:
            return tuple((i,) for i in range(self.neurons()))
        else:
            if len(idxs) != self.neurons():
                raise ValueError(
                    f"""Node: {self}
                    Expected idxs to be size {self.neurons()} but was size {len(idxs)}"""
                )
            return idxs

    def nd_index(self, idxs: List[int], neurons: Optional[int] = None):
        if neurons is None:
            neurons = self.neurons()

        scale = self.dendrites() / neurons

        if scale == 1:
            return idxs
        elif scale > 1:
            out_idxs = []
            for idx in idxs:
                out_idxs.extend([idx + scale * i for i in range(int(scale))])
        else:
            dendrites = self.dendrites()
            out_idxs = [idx for idx in idxs if idx < dendrites]
            assert (
                len(out_idxs) / len(idxs) == scale
            ), f"{self.__class__.__name__}: scale={scale}, dendrites={self.dendrites()}, neurons={neurons}"
            for idx in out_idxs:
                for i in range(int(scale)):
                    assert (idx + scale * i) in idxs

        return out_idxs


class OpNode(ElementNode):
    """Op node — terminates graph traversal because it owns parameters."""

    _module_type = (torch.nn.Module,)
    _data_type = (ModuleData,)
    traverse_node: bool = False
    TYPES: Dict[Union[Type[torch.nn.Module], str], OpNode] = {}

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

    def destroy_outputs(self, idxs: List[int]):
        raise NotImplementedError(
            f"Not implemented for class {self.__class__.__name__}"
        )

    def destroy_inputs(self, idxs: List[int]):
        raise NotImplementedError(
            f"Not implemented for class {self.__class__.__name__}"
        )

    def merge_outputs(self, groups: List[List[int]]) -> List[int]:
        raise NotImplementedError(
            f"Not implemented for class {self.__class__.__name__}"
        )

    def merge_inputs(self, groups: List[List[int]]) -> List[int]:
        raise NotImplementedError(
            f"Not implemented for class {self.__class__.__name__}"
        )

    def nd_index(self, idxs: List[int], neurons: Optional[int] = None):
        return None


# =============================================================================
# Helpers
# =============================================================================


def _redundant_idxs_from_groups(groups: List[List[int]]) -> List[int]:
    """Given groups of indices, return everything except the first of each."""
    out: List[int] = []
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

    def merge_outputs(self, groups: List[List[int]]) -> List[int]:
        """Average weight, bias, running_mean, running_var across each group.
        Stores the merged value into the first member; other members are
        returned as redundant indices."""
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

    def destroy_outputs(self, idxs: List[int]):
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

    def merge_outputs(self, groups: List[List[int]]) -> List[int]:
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

    def destroy_outputs(self, idxs: List[int]):
        layer = self.data.module
        keep_idxs = sorted(set(range(layer.num_channels)) - set(idxs))
        layer.num_channels = layer.num_channels - len(idxs)
        # num_groups must still divide num_channels evenly. If not, downstream
        # behaviour is undefined; warn the caller via assertion.
        assert (
            layer.num_channels % layer.num_groups == 0
        ), "GroupNorm num_channels must remain divisible by num_groups after destroy"

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

    def merge_outputs(self, groups: List[List[int]]) -> List[int]:
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

    def destroy_outputs(self, idxs: List[int]):
        layer = self.data.module
        n = self.dendrites()
        keep_idxs = sorted(set(range(n)) - set(idxs))
        new_n = n - len(idxs)
        layer.normalized_shape = (new_n,) if isinstance(layer.normalized_shape, tuple) else new_n

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
        idxs: Union[Tuple[Tuple[int], ...], Tuple[Tuple[Tuple, ...], ...], None] = None,
    ):
        idxs = super().linked_neurons(idxs)
        dendrites = self.dendrites()
        return tuple(tuple(idxs[i::dendrites]) for i in range(dendrites))


# =============================================================================
# Specialized nodes — Conv (regular + depthwise group)
# =============================================================================


@OpNode.register(torch.nn.modules.conv._ConvNd)
class ConvNode(OpNode):
    def __init__(self, data: Union[GradientData, ModuleData]):
        super().__init__(data)
        assert isinstance(data, ModuleData)
        if data.module.groups != 1:
            raise NodeError

    def dendrites(self):
        return self.data.module.in_channels

    def neurons(self):
        return self.data.module.out_channels

    def merge_outputs(self, groups: List[List[int]]) -> List[int]:
        """Average weights and biases across each group's output channels.
        Linearity rule: the producing-layer weights are *averaged* so the
        merged neuron's output equals the average of the originals."""
        layer = self.data.module
        if layer.transposed:
            out_axis = 1
        else:
            out_axis = 0
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
        _ = out_axis  # documented but not needed once branched above
        return _redundant_idxs_from_groups(groups)

    def merge_inputs(self, groups: List[List[int]]) -> List[int]:
        """Sum input weight columns across each group. Linearity rule:
        the consuming-layer weights are *summed* so the new effective
        weight is `sum_i W_i` instead of `W_i * mean_signal`."""
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

    def destroy_outputs(self, idxs: List[int]):
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

    def destroy_inputs(self, idxs: List[int]):
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

    def __init__(self, data: Union[GradientData, ModuleData]):
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

    def merge_outputs(self, groups: List[List[int]]) -> List[int]:
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

    def destroy_outputs(self, idxs: List[int]):
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

    def merge_outputs(self, groups: List[List[int]]) -> List[int]:
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

    def merge_inputs(self, groups: List[List[int]]) -> List[int]:
        """Sum weight columns across each group's input features."""
        layer = self.data.module
        for group in groups:
            if len(group) < 2:
                continue
            dead = group[0]
            layer.weight.data[:, dead] = layer.weight.data[:, group].sum(dim=1)
        return _redundant_idxs_from_groups(groups)

    def destroy_outputs(self, idxs: List[int]):
        layer = self.data.module
        keep_idxs = sorted(set(range(layer.out_features)) - set(idxs))
        layer.out_features -= len(idxs)
        layer.weight = torch.nn.Parameter(layer.weight.data.clone()[keep_idxs])
        if layer.bias is not None:
            layer.bias = torch.nn.Parameter(layer.bias.data.clone()[keep_idxs])
        return keep_idxs, idxs

    def destroy_inputs(self, idxs: List[int]):
        layer = self.data.module
        keep_idxs = sorted(set(range(layer.in_features)) - set(idxs))
        layer.in_features = layer.in_features - len(idxs)
        layer.weight = torch.nn.Parameter(layer.weight.data.clone()[:, keep_idxs])
        return keep_idxs, idxs


# =============================================================================
# Edges and graph types
# =============================================================================


class Dependency(Edge):
    x: Union[OpNode, ElementNode]
    y: Union[OpNode, ElementNode]
    w: Union[int, float] = 1
    data: Optional[Any] = None
    connect: bool = False

    def __init__(
        self,
        x: Union[OpNode, ElementNode],
        y: Union[OpNode, ElementNode],
        w: Union[int, float] = 1,
        data: Optional[Any] = None,
        connect: bool = False,
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

    def destroy(self, idxs: List[int]):
        """Destroy connected neurons across the graph at ``idxs`` (no merge)."""
        stack_type = Tuple[Union[OpNode, ElementNode], List[int]]

        leaves = self.leaves()
        roots = self.roots()
        process_stack: List[stack_type] = [(leaf, idxs) for leaf in leaves]
        visited: Set[Union[OpNode, ElementNode]] = set()

        destroy_stack: List[stack_type] = []

        while len(process_stack) > 0:
            node, indices = process_stack.pop()
            if node in visited:
                continue
            visited.add(node)

            new_indices = node.nd_index(indices)
            for edge in node.e_in():
                vertex: Union[OpNode, ElementNode] = edge.v[0]
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
            if node not in leaves:
                node.destroy_outputs(indices)
            if node not in roots:
                node.destroy_inputs(indices)

    # Back-compat alias.
    apoptosis = destroy

    def apoptose(self, groups: List[List[int]]):
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

        stack_type = Tuple[Union[OpNode, ElementNode], List[List[int]]]
        leaves = self.leaves()
        roots = self.roots()
        process_stack: List[stack_type] = [(leaf, groups) for leaf in leaves]
        visited: Set[Union[OpNode, ElementNode]] = set()
        apoptose_stack: List[stack_type] = []

        while len(process_stack) > 0:
            node, node_groups = process_stack.pop()
            if node in visited:
                continue
            visited.add(node)

            # Translate indices through reshape/pool nodes if they alter the
            # output-to-input mapping.
            flat = _redundant_idxs_from_groups(node_groups)
            new_flat = node.nd_index(flat)
            if new_flat is not None and new_flat != flat:
                # Heuristic: if nd_index reorders/scales, we conservatively
                # keep the original group structure for the upstream walk.
                # (Most layer types don't alter neuron identity.)
                pass

            for edge in node.e_in():
                vertex: Union[OpNode, ElementNode] = edge.v[0]
                process_stack.append((vertex, node_groups))

            if isinstance(node.data, ModuleData):
                apoptose_stack.append((node, node_groups))

        visited = set()
        for node, node_groups in apoptose_stack:
            if node in visited:
                continue
            visited.add(node)

            if node not in leaves:
                node.merge_outputs(node_groups)
                node.destroy_outputs(_redundant_idxs_from_groups(node_groups))
            if node not in roots:
                node.merge_inputs(node_groups)
                node.destroy_inputs(_redundant_idxs_from_groups(node_groups))

    def linked_neurons(self):
        leaves = self.leaves()
        roots = self.roots()

        if len(list(self.V())) == 1 or leaves[0].dendrites() == roots[0].neurons():
            return [tuple([(i,) for i in range(leaves[0].neurons())])]

        if len(leaves) != 1:
            raise NotImplementedError("Multiple output nodes are not yet supported")

        process_stack = []
        for node in leaves:
            indices = list([(i,) for i in range(node.dendrites())])
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
    _visit_count: Dict[torch.nn.Module, int] = {}
    _grad_trace: Dict[Any, torch.nn.Module] = {}
    _block_hook: bool = True

    def __init__(
        self,
        model: torch.nn.Module,
        inp_tensor: module_io = torch.ones((1, 3, 256, 256)),
    ):
        self.model = model
        self._visit_count = {module: 0 for module in model.modules()}

        hooks = [
            module.register_forward_hook(self)
            for module in model.modules()
            if not list(module.children())
        ]

        self._block_hook = False
        self.model.eval()
        device = next(model.parameters()).device
        out: torch.Tensor = self.model(inp_tensor.to(device))
        self._block_hook = True

        for hook in hooks:
            hook.remove()

        # Trace the network and generate the nodes and edges
        gradients: List[torch.autograd.graph.Node] = [out.grad_fn]
        visited_nodes: List[torch.autograd.graph.Node] = []

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

        nodes: Dict[Any, Union[OpNode, ElementNode]] = {out.grad_fn: node}
        edges: List[Dependency] = []

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

                vertices: List[Vertex] = [
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
                + "should not be directly called."
            )
        self._visit_count[module] += 1
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        self._grad_trace[outputs.grad_fn] = module
