import pytest
import torch

import TensorState as ts  # noqa: N813 -- deliberate package alias
from TensorState import testing as ts_testing
from TensorState.Layers import StateCaptureHook


def test_capture_under_torch_compile():
    """Probes must still capture when the model is torch.compile'd (AIQ-22).

    Regression guard: without ``torch._dynamo.disable`` on the hook, dynamo
    traced past the host-side capture and silently recorded 0 states.
    """
    try:
        torch._dynamo.reset()
        torch.compile(lambda t: t + 1)(torch.zeros(1))
    except Exception:  # noqa: BLE001 -- torch.compile unavailable on this env
        pytest.skip("torch.compile unavailable in this environment")

    torch._dynamo.reset()
    model = ts_testing.small_model("lenet5", num_classes=10)
    ts.build_efficiency_model(model, attach_to=["Conv2dNormActivation"])
    model.eval()
    compiled = torch.compile(model)
    with torch.no_grad():
        compiled(torch.randn(8, 3, 64, 64))

    probes = ts.layers(model)
    assert probes, "no probes attached"
    for name, probe in probes.items():
        assert isinstance(probe, StateCaptureHook)
        assert probe.state_count > 0, (
            f"compiled capture dropped states for {name}: {probe.state_count}"
        )


def test_capture_layers(model, data, capture_states, device, disk_path, benchmark):
    _train, test = data

    model_gen, layer = model
    m = model_gen(num_classes=len(test.dataset.classes))
    if capture_states:
        ts.build_efficiency_model(m, attach_to=[layer], storage_path=disk_path)

    m.to(device)
    m.eval()

    # warmup
    for x, _y in test:
        m(x.to(device))

        break

    # benchmark
    for x, _y in test:
        benchmark(m, x.to(device))

        break
