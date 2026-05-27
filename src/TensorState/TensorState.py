import logging
from collections import OrderedDict

import numpy as np
import torch

from TensorState.Layers import StateCaptureHook

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
    # Extract efficiency values from a model with attached hooks
    if hasattr(efficiencies, "efficiency_layers"):
        efficiencies = [eff.efficiency() for eff in efficiencies.efficiency_layers]
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


def entropy(counts, alpha=1):
    """Calculate the Renyi entropy.

    The Renyi entropy is a general definition of entropy that encompasses
    Shannon's entropy, Hartley (maximum) entropy, and min-entropy. It is defined
    as:

    ``(1-alpha)**-1 * log2( sum(p**alpha) )``

    By default, this method sets alpha=1, which is Shannon's entropy.

    Args:
        counts: Array of counts representing number of times a state is
            observed.
        alpha: Entropy order. Defaults to 1.

    Returns:
        The entropy of the count data.
    """
    num_microstates = counts.sum()
    frequencies = counts / num_microstates
    if alpha == 1:
        entropy = (-frequencies * np.log2(frequencies)).sum()
    else:
        entropy = 1 / (1 - alpha) * np.log2((frequencies**alpha).sum())

    return entropy


def reset_efficiency_model(model):
    """Reset all efficiency layers/hooks in a model.

    This method resets all efficiency layers or hooks in a model, setting the
    ``state_count=0``. This is useful for repeated evaluation of a model
    during a single session.

    Args:
        model: Model to reset
    """
    for layer in model.efficiency_layers:
        layer.reset_states()


def _pt_efficiency_model(
    model,
    attach_to,
    exclude,
    method,
    storage_path,
    memory_device,
    raise_on_capture_error,
):
    model.efficiency_layers = []
    model.state_capture_hooks = []
    # Probes are owned here, off every module's forward path. Adding them
    # as children of the watched module would break container modules
    # (nn.Sequential and friends) whose forward iterates their children.
    model._tensorstate_probes = torch.nn.ModuleDict()

    layer_ids = {
        id(module): (module.__class__.__name__, None, module)
        for module in model.modules()
    }
    layer_ids.update(
        {
            id(module): (module.__class__.__name__, name, module)
            for name, module in model.named_modules()
        }
    )

    for cls_name, mod_name, module in layer_ids.values():
        if (
            cls_name not in attach_to or mod_name in exclude
        ) and mod_name not in attach_to:
            continue

        # ModuleDict keys cannot contain "."; sanitize the qualified name.
        base_key = (mod_name or "root").replace(".", "__")

        # Add pre-hook if requested
        if method in ["before", "both"]:
            efficiency_layer = StateCaptureHook(
                name=str(mod_name) + "_pre_states",
                disk_path=storage_path,
                memory_device=memory_device,
                raise_on_capture_error=raise_on_capture_error,
            )
            model._tensorstate_probes[f"{base_key}_pre"] = efficiency_layer
            model.efficiency_layers.append(efficiency_layer)

            model.state_capture_hooks.append(
                module.register_forward_pre_hook(efficiency_layer._capture)
            )

        if method in ["after", "both"]:
            efficiency_layer = StateCaptureHook(
                name=str(mod_name) + "_post_states",
                disk_path=storage_path,
                memory_device=memory_device,
                raise_on_capture_error=raise_on_capture_error,
            )
            model._tensorstate_probes[f"{base_key}_post"] = efficiency_layer
            model.efficiency_layers.append(efficiency_layer)

            model.state_capture_hooks.append(
                module.register_forward_hook(efficiency_layer._capture)
            )

    return model


def build_efficiency_model(
    model,
    attach_to,
    exclude=None,
    method="after",
    storage_path=None,
    memory_device: str | int = "cpu",
    *,
    raise_on_capture_error: bool = False,
):
    """Attach state capture methods to a neural network.

    This method takes an existing PyTorch model and attaches forward hooks
    to capture the firing states of neural network layers. The model is
    modified in place (hooks attached) and returned for convenience.

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
        storage_path: Path on disk to store states in zarr format. If None,
            states are stored in memory. Defaults to None.
        memory_device: "cpu" or "gpu". When "gpu" and torch.cuda is available,
            the state cache is held on GPU before transferring to main memory.
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
    if hasattr(model, "_tensorstate_probes"):
        del model._tensorstate_probes

    for _name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks = OrderedDict()
            if hasattr(child, "_forward_pre_hooks"):
                child._forward_pre_hooks = OrderedDict()
            remove_state_layers(child)

    return model
