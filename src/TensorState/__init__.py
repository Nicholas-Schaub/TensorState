# noqa: D104
import logging

logging.basicConfig(
    format="%(asctime)s - %(name)-10s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("TensorState")

# Detect cupy
try:
    import cupy  # noqa

    has_cupy = True
except ModuleNotFoundError:
    has_cupy = False

logger.info(f"has_cupy: {has_cupy}")

from .Dependency import ElementNode, ModuleGraph, OpNode  # noqa
from .Layers import StateCaptureHook  # noqa
from .models import LeNet_5, mobilenet_v2  # noqa
from .States import compress_states, decompress_states, sort_states  # noqa
from .TensorState import (  # noqa
    aIQ,
    build_efficiency_model,
    entropy,
    network_efficiency,
    reset_efficiency_model,
    zero_info,
)

__version__ = "0.5.0-dev1"
