import pytest

import TensorState as ts  # noqa: N813 -- deliberate package alias


@pytest.mark.parametrize("weights", ["IMAGENET1K_V1", "IMAGENET1K_V2", None])
def test_mobilenet_v2(data, device, weights):
    _train, test = data

    num_classes = len(test.dataset.classes)

    model = ts.models.mobilenet_v2(num_classes=num_classes, weights=weights)

    model.to(device)
    model.eval()

    for x, _y in test:
        model(x.to(device))

        break
