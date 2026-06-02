import pytest
import torch
import torchvision
import torchvision.datasets as datasets
from torchvision.transforms import Compose, Resize, ToTensor

import TensorState as ts  # noqa: N813 -- deliberate package alias
from TensorState import testing as ts_testing

torch_data = [
    # Default: synthetic, no download, LeNet5/AlexNet-compatible at 64x64.
    "tiny",
    pytest.param("MNIST", marks=pytest.mark.all_data),
    pytest.param("KMNIST", marks=pytest.mark.all_data),
    pytest.param("QMNIST", marks=pytest.mark.all_data),
    pytest.param("EMNIST", marks=pytest.mark.all_data),
    pytest.param("FashionMNIST", marks=pytest.mark.all_data),
    pytest.param("CIFAR10", marks=pytest.mark.all_data),
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
        id="ConvNext",
    ),
    pytest.param(
        (torchvision.models.densenet121, "_DenseBlock"),
        marks=pytest.mark.all_models,
        id="DenseNet121",
    ),
]

compress_backend = [
    pytest.param("numpy", marks=pytest.mark.use_cpu),
    pytest.param("torch", marks=pytest.mark.use_cpu),
    pytest.param("torch_cuda", marks=pytest.mark.use_gpu),
]

decompress_backend = ["numpy"]


def expand_channel(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] == 1:
        x = x.repeat_interleave(3, 0)

    return x


@pytest.fixture(scope="module", params=compress_backend)
def compression(request):
    return request.param


@pytest.fixture(scope="module", params=decompress_backend)
def decompression(request):
    return request.param


def _tiny_loaders(num_classes=10):
    """Synthetic, download-free train/test loaders for the default path.

    Uses 64x64 images so the same loaders feed LeNet5 / AlexNet (which
    need a real spatial extent) as well as the smaller models. The
    datasets carry a ``.classes`` attribute so tests can size model heads
    via ``len(test.dataset.classes)`` exactly as they do for torchvision
    datasets.
    """
    train_ds = ts_testing.tiny_dataset(
        n=256, channels=3, size=64, num_classes=num_classes, seed=0
    )
    test_ds = ts_testing.tiny_dataset(
        n=64, channels=3, size=64, num_classes=num_classes, seed=1
    )
    train_ds.classes = list(range(num_classes))  # ty: ignore[unresolved-attribute]  # fixture injects torchvision-style .classes attribute
    test_ds.classes = list(range(num_classes))  # ty: ignore[unresolved-attribute]  # fixture injects torchvision-style .classes attribute
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=64)
    return train_dl, test_dl


@pytest.fixture(scope="module", params=torch_data)
def data(request):
    name = request.param

    if name == "tiny":
        return _tiny_loaders()

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
    params=[
        pytest.param("cuda", marks=pytest.mark.use_gpu),
        pytest.param("cpu", marks=pytest.mark.use_cpu),
    ],
)
def device(request):
    return request.param


@pytest.fixture(params=[True, False])
def capture_states(request):
    return request.param


@pytest.fixture(params=torch_models)
def model(request):
    model, layer = request.param

    return model, layer


@pytest.fixture(params=[None, "disk"])
def disk_path(request, tmp_path_factory):
    if request.param is None:
        return None

    # Use pytest's tmp_path_factory so each parametrized case gets a unique,
    # auto-cleaned directory. This avoids the FileExistsError that occurred
    # when reusing a fixed "./states_<worker>" path across cases.
    return tmp_path_factory.mktemp("states")


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
