import logging

logging.basicConfig(
    format="%(asctime)s - %(name)-10s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("TensorState")

from .core import (  # noqa
    aIQ,
    attach,
    build_efficiency_model,
    efficiency,
    entropy,
    layer,
    layers,
    match,
    network_efficiency,
    remove_state_layers,
    reset_efficiency_model,
    zero_info,
)
from .dependency import ElementNode, ModuleGraph, OpNode  # noqa
from .layers import Probe, StateCaptureHook  # noqa
from .models import LeNet_5, mobilenet_v2  # noqa
from .states import compress_states, decompress_states, sort_states  # noqa

__version__ = "0.5.0.dev1"
