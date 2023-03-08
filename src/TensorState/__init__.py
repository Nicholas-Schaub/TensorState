# noqa: D104
import logging

logging.basicConfig(
    format="%(asctime)s - %(name)-10s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("TensorState")

# Detect cupy
try:
    pass

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

__version__ = "0.3.0"
