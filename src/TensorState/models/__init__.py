# noqa
try:
    import torch  # noqa
    import torchvision  # noqa

    from TensorState.models.LeNet import LeNet_5  # noqa
    from TensorState.models.MobileNetV2 import mobilenet_v2  # noqa

except ImportError:
    pass
