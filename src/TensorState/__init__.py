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

# Detect PyTorch
try:
    pass

    has_torch = True
except ModuleNotFoundError:
    has_torch = False

logger.info(f"has_torch: {has_torch}")

# Detect Tensorflow
try:
    pass

    has_tf = True
except ModuleNotFoundError:
    has_tf = False

logger.info(f"has_tf: {has_tf}")

from .Layers import StateCapture, StateCaptureHook  # noqa
from .models import LeNet_5  # noqa
from .States import compress_states, decompress_states, sort_states  # noqa
from .TensorState import (  # noqa
    aIQ,
    build_efficiency_model,
    entropy,
    network_efficiency,
    reset_efficiency_model,
)

__version__ = "0.4.0-dev1"
