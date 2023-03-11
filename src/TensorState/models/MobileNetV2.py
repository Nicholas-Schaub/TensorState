import torch
import torchvision.models
from torchvision.ops import Conv2dNormActivation


def mobilenet_v2(num_classes=1000, **kwargs):
    """Modify mobilenet to use Conv2dNormActivation layers.

    The purpose of this function is to modify the stock torchvision MobileNetV2
    model to rely more on Conv2dNormActivation so that attaching state capture
    layers is simplified.

    This also modifies the model so that when pre-trained weights are specified,
    and num_classes is not 1000, the classification head is modified to support
    a different number of classes.

    Returns:
        A modified torchvision.models.MobileNetV2 model.
    """
    model = torchvision.models.mobilenet_v2(**kwargs)

    for module in model._modules["features"]:
        if module.__class__.__name__ == "InvertedResidual":
            m = module._modules["conv"]
            keys = list(m._modules.keys())
            conv = m._modules.pop(keys[-2])
            bn = m._modules.pop(keys[-1])

            in_chan = conv.weight.shape[1]
            out_chan = conv.weight.shape[0]
            kernel_size = conv.weight.shape[2:]
            stride = conv.stride
            padding = conv.padding
            bias = conv.bias

            new_layer = Conv2dNormActivation(
                in_channels=in_chan,
                out_channels=out_chan,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                activation_layer=None,
                bias=bias,
            )

            new_layer[0] = conv
            new_layer[1] = bn
            m.append(new_layer)

    if num_classes != 1000:
        dropout = 0.2 if "dropout" not in kwargs else kwargs["dropout"]
        linear = torch.nn.Linear(model.last_channel, num_classes)
        torch.nn.init.normal_(linear.weight, 0, 0.01)
        torch.nn.init.zeros_(linear.bias)
        model._modules["classifier"] = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            linear,
        )

    return model
