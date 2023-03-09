from pathlib import Path
from shutil import rmtree

import pytest
import torch
import torchvision
import torchvision.datasets as datasets
from torchvision.transforms import Compose, Resize, ToTensor

import TensorState as ts

torch_data = [
    "MNIST",
    pytest.param("KMNIST", marks=pytest.mark.all_data),
    pytest.param("QMNIST", marks=pytest.mark.all_data),
    pytest.param("EMNIST", marks=pytest.mark.all_data),
    pytest.param("FashionMNIST", marks=pytest.mark.all_data),
    "CIFAR10",
    pytest.param("CIFAR100", marks=pytest.mark.all_data),
]


torch_models = [
    pytest.param(
        (ts.models.LeNet_5, "Conv2dNormActivation"),
        id="LeNet5",
    ),
    pytest.param(
        (torchvision.models.AlexNet, "Conv2dNormActivation"),
        id="AlexNet",
    ),
    pytest.param(
        (torchvision.models.mobilenet_v2, "Conv2dNormActivation"),
        marks=pytest.mark.all_models,
        id="MobileNetV2",
    ),
    pytest.param(
        (torchvision.models.convnext_base, "CNBlock"),
        marks=pytest.mark.all_models,
    ),
    pytest.param(
        (torchvision.models.densenet121, "_DenseBlock"),
        marks=pytest.mark.all_models,
    ),
]


def expand_channel(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] == 1:
        x = x.repeat_interleave(3, 0)

    return x


@pytest.fixture(scope="module", params=torch_data)
def data(request):
    name = request.param

    """Create the data sets"""
    kwargs = {}
    if name == "EMNIST":
        kwargs.update({"split": "balanced"})
    train_dataset: torch.utils.data.Dataset = getattr(datasets, name)(
        root=".data",
        train=True,
        transform=Compose([ToTensor(), Resize((64, 64)), expand_channel]),
        download=True,
        **kwargs,
    )
    test_dataset: torch.utils.data.Dataset = getattr(datasets, name)(
        root=".data",
        train=False,
        transform=Compose([ToTensor(), Resize((64, 64)), expand_channel]),
        download=True,
        **kwargs,
    )

    """ Create the data loaders """
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=200, num_workers=4)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=200, num_workers=4)

    return train_dl, test_dl


@pytest.fixture(
    scope="function",
    params=[
        pytest.param("cuda", marks=pytest.mark.use_gpu),
        pytest.param("cpu", marks=pytest.mark.use_cpu),
    ],
)
def device(request):
    return request.param


@pytest.fixture(scope="function", params=[True, False])
def capture_states(request):
    return request.param


@pytest.fixture(scope="function", params=torch_models)
def model(request):
    model, layer = request.param

    return model, layer


@pytest.fixture(params=[None, Path("./states")], autouse=True)
def disk_path(request, worker_id):
    path: Path = request.param

    if path is not None:
        path = path.with_name(path.name + f"_{worker_id}")
        path.mkdir()

    yield path

    if path is not None:
        rmtree(path)


def pytest_addoption(parser):
    parser.addoption(
        "--all-models",
        action="store_true",
        default=False,
        help="run tests on all models",
    )
    parser.addoption(
        "--all-data", action="store_true", default=False, help="run tests on all data"
    )
    parser.addoption(
        "--all",
        action="store_true",
        default=False,
        help="run all tests (this takes awhile)",
    )
    parser.addoption(
        "--use-gpu",
        action="store_true",
        default=False,
        help="run all tests on gpu in addition to cpu",
    )
    parser.addoption(
        "--no-cpu",
        action="store_true",
        default=False,
        help="Only run tests on gpu",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--all"):
        return

    if not config.getoption("--all-models"):
        skip_model = pytest.mark.skip(reason="need --all-models option to run")
        for item in items:
            if "all_models" in item.keywords:
                item.add_marker(skip_model)

    if not config.getoption("--all-data"):
        skip_data = pytest.mark.skip(reason="need --all-data option to run")
        for item in items:
            if "all_data" in item.keywords:
                item.add_marker(skip_data)

    if config.getoption("--no-cpu"):
        skip_data = pytest.mark.skip(reason="--no-cpu was used")
        for item in items:
            if "use_cpu" in item.keywords:
                item.add_marker(skip_data)

        return

    if not config.getoption("--use-gpu"):
        skip_data = pytest.mark.skip(reason="need --use-gpu option to run")
        for item in items:
            if "use_gpu" in item.keywords:
                item.add_marker(skip_data)
