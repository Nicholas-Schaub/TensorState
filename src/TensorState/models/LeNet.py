import logging
from typing import List, Tuple, Union

import torch
import torchvision

logger = logging.getLogger(__name__)


class LeNet_5(torch.nn.Module):
    """LeNet 5 model."""

    def __init__(
        self,
        layers: Union[
            int,
            Tuple[Union[int, List[int]], Union[int, List[int]], Union[int, List[int]]],
        ] = (64, 64, 64),
        batch_norm=True,
        residual=False,
        num_classes=10,
    ):
        """Initialize LeNet 5 model.

        Create a parametrized version of LeNet 5. The parameters allow a simple
        definition of number of layers, number of neurons in each layer, whether
        to include batch_norm and/or residual connections, and initialize with
        a specified number of classes.

        Args:
            layers: An int or 3-tuple, where each element of the tuple is the
                number of neurons in each of the three layers. If any of the
                elements of a tuple is a list of integers, then multiple layers
                are created. Defaults to (64, 64, 64).
            batch_norm: Use batch normalization when True. Defaults to True.
            residual: Use residual connections when True. Defaults to False.
            num_classes: Number of classes. Defaults to 10.
        """
        super().__init__()

        self.use_res_connect = residual

        if isinstance(layers, tuple):
            assert len(layers) == 3

            new_layers: List[List[int]] = []
            for layer in layers:
                if isinstance(layer, int):
                    new_layers.append([layer])
                elif isinstance(layer, list):
                    new_layers.append(layer)
                else:
                    raise TypeError
            tuple_layers = tuple(new_layers[:3])
        else:
            assert isinstance(layers, int)
            tuple_layers = ([layers],) * 3

        blocks: List[torchvision.ops.Conv2dNormActivation] = []
        in_chan = 3
        for nl, layer in enumerate(tuple_layers):
            for nn, num_neurons in enumerate(layer):
                is_last = nn == len(layer)
                blocks.append(
                    torchvision.ops.Conv2dNormActivation(
                        in_channels=in_chan,
                        out_channels=num_neurons,
                        kernel_size=4 if nl == 2 else 5,
                        activation_layer=torch.nn.ELU,
                        padding=(0, 0),
                        bias=False,
                        norm_layer=torch.nn.BatchNorm2d if batch_norm else None,
                    )
                )
                if not is_last and nl != 2:
                    blocks[-1].append(torch.nn.MaxPool2d(2, stride=2))
                in_chan = num_neurons

        self.features = torch.nn.Sequential(*blocks)

        # Prediction
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(in_chan, num_classes),
            torch.nn.BatchNorm1d(num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):  # noqa
        if self.use_res_connect:
            for n, layer in enumerate(self.features):
                if n == 1:
                    x = layer(x) + torch.nn.MaxPool2d(2, stride=2)(x[..., 2:-2, 2:-2])
                else:
                    x = layer(x)

        else:
            x = self.features(x)

        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))

        x = self.classifier(x)

        return x
