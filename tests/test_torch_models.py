import pytest

import TensorState as ts


@pytest.mark.parametrize("weights", ["IMAGENET1K_V1", "IMAGENET1K_V2", None])
def test_mobilenet_v2(data, device, weights):
    train, test = data

    num_classes = len(test.dataset.classes)

    model = ts.models.mobilenet_v2(num_classes=num_classes, weights=weights)

    model.to(device)
    model.eval()

    for x, y in test:
        z = model(x.to(device))
