from __future__ import absolute_import, unicode_literals
import logging

logging.basicConfig(format='%(asctime)s - %(name)-10s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger('TensorState')

# Detect cupy
try:
    import cupy
    has_cupy = True
except ModuleNotFoundError:
    has_cupy = False

logger.info('has_cupy: {}'.format(has_cupy))

# Detect PyTorch
try:
    import torch
    has_torch = True
except ModuleNotFoundError:
    has_torch = False
    
logger.info('has_torch: {}'.format(has_torch))
    
# Detect Tensorflow
try:
    import tensorflow
    has_tf = True
except ModuleNotFoundError:
    has_tf = False
    
logger.info('has_tf: {}'.format(has_tf))

from .TensorState import reset_efficiency_model, build_efficiency_model, \
                         entropy, aIQ, network_efficiency
from .Layers import *
from .States import *
from ._TensorState import *