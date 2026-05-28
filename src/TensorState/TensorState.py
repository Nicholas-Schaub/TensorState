import logging
import re
import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np
import torch

from TensorState.Layers import Probe, StateCaptureHook

logging.basicConfig(
    format="%(asctime)s - %(name)-10s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("TensorState")
logger.setLevel(logging.WARNING)


def zero_info(states):
    """Get index of zero information neurons.

    This function evaluates state space to find zero information neurons, or
    groups of neurons that fire in perfect sync. When two neurons fire in
    perfect sync, the neurons can effectively be thought of as a single
    neuron for the purposes of calculating entropy.

    In addition to neurons that fire in perfect sync, there are two special
    cases of neurons not contributing to entropy: always on and always off
    neurons.

    Args:
        states: An array of unique states.

    Returns:
        A list of lists, where each element in the list is a list of neurons
            that form a group of zero information neurons. There will always be
            at least two elements to the list: always off and always on. If
            there are no always off or always on neurons, the lists will be
            empty.
    """
    # Find the correlation between neurons
    corr = np.corrcoef(states.T)

    # Find off and on neurons
    off_neurons = np.argwhere(np.isnan(corr).all(axis=0) & ~states[0]).flatten()
    on_neurons = np.argwhere(np.isnan(corr).all(axis=0) & states[0]).flatten()

    # Make the threshold slightly less than one to account for delta issues
    threshold = 1.0 - 1.0 / states.shape[0] ** 2
    adjacency = np.abs(corr) >= threshold

    indices = np.argwhere(adjacency.any(axis=0)).flatten()

    corr_neurons = [off_neurons.tolist(), on_neurons.tolist()]

    # Build a list of zero information neuron groups
    while indices.size > 0:
        corr_neurons.append(np.argwhere(adjacency[indices[0]]).flatten().tolist())
        adjacency[corr_neurons[-1]] = False
        if len(corr_neurons[-1]) == 1:
            corr_neurons.pop()
        indices = np.argwhere(adjacency.any(axis=1)).flatten()

    return corr_neurons


def network_efficiency(efficiencies):
    """Calculate the network efficiency.

    This method calculates the neural network efficiency, defined as the
    geometric mean of the efficiency values calculated for the network.

    Args:
        efficiencies: A list of efficiency values (floats), or a PyTorch /
            Lightning module with attached state-capture hooks.

    Returns:
        The network efficiency
    """
    # Extract efficiency values from a model with attached probes
    if isinstance(efficiencies, torch.nn.Module):
        efficiencies = [p.efficiency() for p in layers(efficiencies).values()]
    assert isinstance(efficiencies, list), (
        "Input must be a list or a module with attached state-capture hooks"
    )

    # If the length of efficiencies is 0, return None and warn the user
    if len(efficiencies) == 0:
        logger.warning(
            "List of efficiency values is empty. Verify input or model "
            "input to network_efficiency."
        )
        return None

    # Geometric mean of efficiencies
    return np.exp(sum(np.log(eff) for eff in efficiencies) / len(efficiencies))


def aIQ(net_efficiency, accuracy, weight):  # noqa: N802 -- public API: aIQ metric
    """Calculate the artificial intelligence quotient.

    The artificial intelligence quotient (aIQ) is a simple metric to report a
    balance of neural network efficiency and task performance. Although not
    required, it is assumed that the accuracy argument is a float ranging from
    0.0-1.0, with 1.0 meaning more accurate.

    aIQ = (net_efficiency * accuracy ** weight) ** (1/(weight+1))

    The weight argument is an integer, with higher values giving more weight to
    the accuracy of the model.

    Args:
        net_efficiency: A float ranging from 0.0-1.0
        accuracy: A float ranging from 0.0-1.0
        weight: An integer with value >=1

    Raises:
        Raised if weight <= 0

    Returns:
        The artificial intelligence quotient
    """
    if weight <= 0 or not isinstance(weight, int):
        raise ValueError("aIQ weight must be an integer greater than 0.")
    return np.power(accuracy**weight * net_efficiency, 1 / (weight + 1))


def entropy(model_or_counts, alpha=1, name=None):
    """Calculate the Renyi entropy.

    The Renyi entropy is a general definition of entropy that encompasses
    Shannon's entropy, Hartley (maximum) entropy, and min-entropy. It is defined
    as:

    ``(1-alpha)**-1 * log2( sum(p**alpha) )``

    By default, this method sets alpha=1, which is Shannon's entropy.

    This function has two forms:

    - **Inspection form**: when the first argument is a model (a
      ``torch.nn.Module`` with attached probes), returns a
      ``dict[str, float]`` mapping probe name to layer entropy. Pass
      ``name=`` to get the entropy of a single layer as a ``float``.
    - **Count form** (legacy): when the first argument is a count array,
      returns the entropy of those counts.

    Args:
        model_or_counts: A model with attached probes, or an array of counts
            representing the number of times each state is observed.
        alpha: Entropy order. Defaults to 1.
        name: When the first argument is a model, restrict to the single probe
            with this name and return a ``float``. Invalid for the count form.

    Returns:
        A ``dict[str, float]`` (model form), a ``float`` (model form with
        ``name=`` or count form).
    """
    if isinstance(model_or_counts, torch.nn.Module):
        if name is not None:
            return layer(model_or_counts, name).entropy(alpha)
        return {n: p.entropy(alpha) for n, p in _iter_probes(model_or_counts)}

    if name is not None:
        raise TypeError("name= is only valid when the first argument is a model.")

    counts = model_or_counts
    num_microstates = counts.sum()
    frequencies = counts / num_microstates
    if alpha == 1:
        entropy = (-frequencies * np.log2(frequencies)).sum()
    else:
        entropy = 1 / (1 - alpha) * np.log2((frequencies**alpha).sum())

    return entropy


def _iter_probes(model) -> Iterator[tuple[str, Probe]]:
    """Yield ``(clean_name, probe)`` for every probe attached to ``model``.

    Walks ``model.named_modules()`` and filters for :class:`Probe`. The key is
    the probe's own ``.name`` with the ``_pre_states``/``_post_states`` suffix
    reduced to ``_pre``/``_post`` so callers key on the watched-layer name and
    hook side (e.g. ``backbone.conv1_post``) rather than the internal container
    key. Yields in attach order.
    """
    seen = set()
    for module in model.modules():
        if isinstance(module, Probe) and id(module) not in seen:
            seen.add(id(module))
            raw = getattr(module, "name", None) or "probe"
            clean = raw[: -len("_states")] if raw.endswith("_states") else raw
            yield clean, module


def layers(model) -> dict[str, Probe]:
    """Probes attached to a model, keyed by clean probe name.

    Args:
        model: A model that has been through ``build_efficiency_model``.

    Returns:
        ``dict[str, Probe]`` in attach order. Compose with the metric APIs via
        ``{n: p.entropy() for n, p in ts.layers(m).items()}``; the keys line up
        with the dicts returned by :func:`entropy` and :func:`efficiency`.
    """
    return dict(_iter_probes(model))


def layer(model, name: str) -> Probe:
    """Return the single attached probe named ``name``.

    Args:
        model: A model that has been through ``build_efficiency_model``.
        name: Clean probe name (see :func:`layers`).

    Raises:
        KeyError: If no probe with that name is attached.

    Returns:
        The matching :class:`Probe`.
    """
    probes = layers(model)
    try:
        return probes[name]
    except KeyError:
        raise KeyError(
            f"No probe named {name!r}. Attached probes: {sorted(probes)}"
        ) from None


def efficiency(model, alpha1=1, alpha2=None, reduce=None, name=None):
    """Per-layer (or reduced) efficiency for a model with attached probes.

    Args:
        model: A model that has been through ``build_efficiency_model``.
        alpha1: Order of Renyi entropy in the numerator. Defaults to 1.
        alpha2: Order of Renyi entropy in the denominator. Defaults to None.
        reduce: If ``None``, return a ``dict[str, float]`` of per-layer
            efficiencies. If ``"geomean"``, return the geometric mean across
            layers as a ``float`` (the network efficiency).
        name: Restrict to the single probe with this name and return a
            ``float``. Mutually exclusive with ``reduce``.

    Raises:
        ValueError: If ``reduce`` is neither ``None`` nor ``"geomean"``.

    Returns:
        ``dict[str, float]`` (default), or a ``float`` (with ``name=`` or
        ``reduce="geomean"``).
    """
    if name is not None:
        return layer(model, name).efficiency(alpha1, alpha2)

    per_layer = {n: p.efficiency(alpha1, alpha2) for n, p in _iter_probes(model)}

    if reduce is None:
        return per_layer
    if reduce == "geomean":
        return network_efficiency(list(per_layer.values()))
    raise ValueError(f"Unknown reduce={reduce!r}; expected 'geomean' or None.")


class _DeprecatedProbeList(list):
    """Back-compat shim for the retired ``model.efficiency_layers`` list.

    Iterating, indexing, or taking the length emits a ``DeprecationWarning``
    and resolves to the live probes via :func:`layers`. Slated for removal one
    release after introduction; new code should use ``ts.layers(model)``.
    """

    def __init__(self, model):
        super().__init__()
        self._model = model

    def _resolve(self) -> list[Probe]:
        warnings.warn(
            "model.efficiency_layers is deprecated; use TensorState.layers(model).",
            DeprecationWarning,
            stacklevel=3,
        )
        return list(layers(self._model).values())

    def __iter__(self):
        return iter(self._resolve())

    def __len__(self):
        return len(self._resolve())

    def __getitem__(self, index):
        return self._resolve()[index]


def reset_efficiency_model(model):
    """Reset all efficiency layers/hooks in a model.

    This method resets all efficiency layers or hooks in a model, setting the
    ``state_count=0``. This is useful for repeated evaluation of a model
    during a single session.

    Args:
        model: Model to reset
    """
    for probe in layers(model).values():
        probe.reset_states()


class _Matcher:
    """Composable ``(name, module) -> bool`` predicate.

    Build via :func:`match`; compose with ``|`` (union), ``&`` (intersection),
    and ``~`` (negation). Bare callables work anywhere a ``_Matcher`` does;
    wrapping them is only needed if you want the operator composition.
    """

    def __init__(self, fn: Callable[[str, torch.nn.Module], bool]):
        self._fn = fn

    def __call__(self, name: str, module: torch.nn.Module) -> bool:
        return self._fn(name, module)

    def __or__(self, other: Callable[[str, torch.nn.Module], bool]) -> "_Matcher":
        return _Matcher(lambda n, m: self(n, m) or other(n, m))

    def __and__(self, other: Callable[[str, torch.nn.Module], bool]) -> "_Matcher":
        return _Matcher(lambda n, m: self(n, m) and other(n, m))

    def __invert__(self) -> "_Matcher":
        return _Matcher(lambda n, m: not self(n, m))


def match(
    *,
    types: type | tuple[type, ...] | None = None,
    name: str | re.Pattern | Iterable[str] | None = None,
    predicate: Callable[[str, torch.nn.Module], bool] | None = None,
) -> _Matcher:
    """Compose a ``(name, module) -> bool`` predicate from optional checks.

    All supplied conditions must hold (logical AND). Compose multiple matchers
    via ``|`` / ``&`` / ``~``.

    Args:
        types: An ``nn.Module`` subclass or tuple of subclasses. The module
            must pass ``isinstance``.
        name: A regex (``str`` or compiled ``re.Pattern``) or an iterable of
            exact qualified names. ``str`` is treated as regex
            (``re.search``); pass a compiled pattern to be explicit.
        predicate: An arbitrary ``(name, module) -> bool`` callable for any
            checks the structured kwargs can't express.

    Returns:
        A :class:`_Matcher`.
    """
    type_tuple: tuple[type, ...] | None
    if types is None:
        type_tuple = None
    elif isinstance(types, type):
        type_tuple = (types,)
    else:
        type_tuple = tuple(types)

    name_pat: re.Pattern | None = None
    name_set: set[str] | None = None
    if isinstance(name, str):
        name_pat = re.compile(name)
    elif isinstance(name, re.Pattern):
        name_pat = name
    elif name is not None:
        name_set = set(name)

    def fn(n: str, m: torch.nn.Module) -> bool:
        if type_tuple is not None and not isinstance(m, type_tuple):
            return False
        if name_pat is not None and name_pat.search(n) is None:
            return False
        if name_set is not None and n not in name_set:
            return False
        if predicate is not None and not predicate(n, m):
            return False
        return True

    return _Matcher(fn)


def _iter_targets(
    model: torch.nn.Module,
    where: Callable[[str, torch.nn.Module], bool],
) -> list[tuple[str, torch.nn.Module]]:
    """Walk ``model.named_modules()`` and return matched ``(name, module)`` pairs."""
    seen: set[int] = set()
    targets: list[tuple[str, torch.nn.Module]] = []
    for n, m in model.named_modules():
        if id(m) in seen:
            continue
        seen.add(id(m))
        if where(n, m):
            targets.append((n, m))
    return targets


def _attach_probes(
    model: torch.nn.Module,
    targets: list[tuple[str, torch.nn.Module]],
    *,
    when: str,
    storage_path: str | Path | None,
    memory_device: str | int,
    memory_limit: str | None,
    raise_on_capture_error: bool,
) -> torch.nn.Module:
    """Attach state-capture probes to a fixed list of target modules.

    Resets ``model.state_capture_hooks`` and ``model._tensorstate_probes``.
    Probes live in the top-level ``_tensorstate_probes`` ``ModuleDict`` so
    container modules (``nn.Sequential`` and friends) — whose ``forward``
    iterates their children — aren't disturbed by adding a child probe.
    """
    if when not in ("before", "after", "both"):
        raise ValueError(f"when must be 'before' | 'after' | 'both', got {when!r}")

    model.state_capture_hooks = []
    model._tensorstate_probes = torch.nn.ModuleDict()

    for mod_name, module in targets:
        # ModuleDict keys cannot contain "."; sanitize the qualified name.
        base_key = (mod_name or "root").replace(".", "__")

        if when in ("before", "both"):
            probe = StateCaptureHook(
                name=str(mod_name) + "_pre_states",
                disk_path=storage_path,
                memory_device=memory_device,
                raise_on_capture_error=raise_on_capture_error,
                memory_limit=memory_limit,
            )
            model._tensorstate_probes[f"{base_key}_pre"] = probe
            model.state_capture_hooks.append(
                module.register_forward_pre_hook(probe._capture)
            )

        if when in ("after", "both"):
            probe = StateCaptureHook(
                name=str(mod_name) + "_post_states",
                disk_path=storage_path,
                memory_device=memory_device,
                raise_on_capture_error=raise_on_capture_error,
                memory_limit=memory_limit,
            )
            model._tensorstate_probes[f"{base_key}_post"] = probe
            model.state_capture_hooks.append(
                module.register_forward_hook(probe._capture)
            )

    # Deprecated back-compat handle; new code should use TensorState.layers().
    model.efficiency_layers = _DeprecatedProbeList(model)
    return model


def attach(
    model: torch.nn.Module,
    where: Callable[[str, torch.nn.Module], bool],
    *,
    when: str = "after",
    storage_path: str | Path | None = None,
    memory_device: str | int = "cpu",
    memory_limit: str | None = None,
    raise_on_capture_error: bool = False,
) -> torch.nn.Module:
    """Attach state-capture probes to layers selected by a predicate.

    ``where(name, module)`` is called for each named submodule of ``model``;
    matching modules get a probe attached. Build matchers with :func:`match`,
    which compose via ``|``, ``&``, and ``~``::

        ts.attach(
            model,
            where=ts.match(types=(nn.Conv2d, nn.Linear))
                  & ts.match(name=r"^backbone\\."),
            when="after",
            memory_limit="4GB",
        )

    This is the new public attach API; :func:`build_efficiency_model` is a
    legacy shim that translates its string-list ``attach_to`` argument to a
    predicate and delegates here.

    Args:
        model: A PyTorch ``nn.Module``.
        where: A ``(name, module) -> bool`` callable (use :func:`match`).
        when: ``"before"``, ``"after"``, or ``"both"``.
        storage_path: Directory under which to back the DuckDB store with an
            on-disk database. In-memory if ``None``.
        memory_device: ``"cpu"`` or ``"gpu"`` — controls the GPU-resident
            compressed-state cache.
        memory_limit: DuckDB ``memory_limit`` PRAGMA value (e.g. ``"4GB"``).
        raise_on_capture_error: When ``True``, a capture failure aborts the
            next capture call by re-raising the stored error.

    Returns:
        ``model`` (modified in place) with probes attached.
    """
    targets = _iter_targets(model, where)
    return _attach_probes(
        model,
        targets,
        when=when,
        storage_path=storage_path,
        memory_device=memory_device,
        memory_limit=memory_limit,
        raise_on_capture_error=raise_on_capture_error,
    )


def _pt_efficiency_model(
    model,
    attach_to,
    exclude,
    method,
    storage_path,
    memory_device,
    raise_on_capture_error,
    memory_limit,
):
    # Translate the legacy attach_to / exclude string-list API to a predicate
    # and reuse the shared _attach_probes machinery. Preserves the historical
    # selection semantics: a module matches if its class name OR its qualified
    # name is in attach_to (and its name is not in exclude).
    attach_set = set(attach_to) if not isinstance(attach_to, str) else {attach_to}
    exclude_set = set(exclude)

    def legacy_where(n: str, m: torch.nn.Module) -> bool:
        if n in exclude_set:
            return False
        return m.__class__.__name__ in attach_set or n in attach_set

    return _attach_probes(
        model,
        _iter_targets(model, legacy_where),
        when=method,
        storage_path=storage_path,
        memory_device=memory_device,
        memory_limit=memory_limit,
        raise_on_capture_error=raise_on_capture_error,
    )


def build_efficiency_model(
    model,
    attach_to,
    exclude=None,
    method="after",
    storage_path=None,
    memory_device: str | int = "cpu",
    *,
    memory_limit: str | None = None,
    raise_on_capture_error: bool = False,
):
    """Attach state capture methods to a neural network.

    This method takes an existing PyTorch model and attaches forward hooks
    to capture the firing states of neural network layers. The model is
    modified in place (hooks attached) and returned for convenience.

    Captured states are kept in a DuckDB-backed store, one bit-packed
    microstate per row. The store is held in memory by default; pass
    ``storage_path`` to back it with an on-disk DuckDB database, and/or
    ``memory_limit`` to cap resident memory (DuckDB spills to a temp dir
    when it would exceed the cap).

    Args:
        model: A PyTorch ``nn.Module`` or Lightning ``LightningModule``.
        attach_to: List of strings indicating the types of layers to attach to.
            Names of layers can also be specified to attach StateCapture to
            specific layers
        exclude: List of strings indicating the names of layers to not attach
            StateCapture layers to. This will override the attach_to keyword, so
            that a Conv2D layer with the name specified by exclude will not have
            a StateCapture layer attached to it. Defaults to [].
        method: The location to attach the StateCapture layer to. Must be one of
            ['before','after','both']. Defaults to 'after'.
        storage_path: Directory under which to back the state store with an
            on-disk DuckDB database. If None (the default), the store is held
            in memory.
        memory_device: "cpu" or "gpu". When "gpu" and torch.cuda is available,
            the state cache is held on GPU before transferring to main memory.
        memory_limit: DuckDB ``memory_limit`` PRAGMA value, e.g. ``"4GB"``.
            When the resident state table grows past this, DuckDB spills to a
            temp directory. ``None`` (the default) leaves DuckDB's own default
            in place.
        raise_on_capture_error: When True, a capture failure aborts the next
            forward pass through a probe by re-raising the stored error. When
            False (the default), capture failures are logged when they occur
            and re-raised only when results are read. Defaults to False.

    Returns:
        model: The same model with state-capture hooks attached.
    """
    if exclude is None:
        exclude = []
    class_module = {cls.__module__: cls.__name__ for cls in model.__class__.__bases__}

    # Validate input arguments
    assert isinstance(attach_to, (list, str))
    assert len(attach_to) > 0
    assert method in [
        "before",
        "after",
        "both",
    ], "Method must be one of [before,after,both]"

    if isinstance(exclude, str):
        exclude = [exclude]
    assert isinstance(exclude, list)

    if (
        class_module.get("torch.nn.modules.module") == "Module"
        or class_module.get("lightning.pytorch.core.module") == "LightningModule"
    ):
        new_model = _pt_efficiency_model(
            model,
            attach_to,
            exclude,
            method,
            storage_path,
            memory_device,
            raise_on_capture_error,
            memory_limit,
        )
    else:
        raise TypeError(
            "build_efficiency_model only supports PyTorch nn.Module and "
            "Lightning LightningModule instances."
        )

    return new_model


def remove_state_layers(model) -> None:
    """Remove state capture layers.

    Note:
        Currently only works with PyTorch.

    Args:
        model: The model to remove hooks from.

    Returns:
        A model with state capture layers removed.
    """
    if hasattr(model, "state_capture_hooks"):
        for hook in model.state_capture_hooks:
            hook.remove()
        del model.state_capture_hooks
        del model.efficiency_layers

    # Drop the probe container (probes live here, off the forward path).
    # Close each probe's state store first so on-disk DuckDB files are released.
    if hasattr(model, "_tensorstate_probes"):
        for probe in model._tensorstate_probes.values():
            store = getattr(probe, "_store", None)
            if store is not None:
                store.close()
        del model._tensorstate_probes

    for _name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks = OrderedDict()
            if hasattr(child, "_forward_pre_hooks"):
                child._forward_pre_hooks = OrderedDict()
            remove_state_layers(child)

    return model
