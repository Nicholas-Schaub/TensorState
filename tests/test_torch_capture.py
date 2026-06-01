import numpy as np
import pytest
import torch

import TensorState as ts  # noqa: N813 -- deliberate package alias
from TensorState import testing as ts_testing
from TensorState.layers import StateCaptureHook


def test_build_efficiency_model_threads_memory_limit_and_to_arrow():
    """memory_limit reaches the store; probes expose a to_arrow() handle (AIQ-39)."""
    model = ts_testing.small_model("lenet5", num_classes=10)
    ts.build_efficiency_model(
        model, attach_to=["Conv2dNormActivation"], memory_limit="1GB"
    )
    model.eval()
    with torch.no_grad():
        model(torch.randn(4, 3, 64, 64))

    for probe in ts.layers(model).values():
        assert isinstance(probe, StateCaptureHook)
        # memory_limit flowed all the way to the probe.
        assert probe._memory_limit == "1GB"
        # Arrow handle materializes a Table with the expected row count. The
        # DuckDB schema is now (step_id BIGINT, s BLOB) -- both columns must
        # be present and row count must match state_count.
        tbl = probe.to_arrow()
        assert tbl.num_rows == probe.state_count
        assert {"step_id", "s"} <= set(tbl.schema.names)


def test_duckdb_store_counts_match_numpy():
    """DuckDB GROUP BY counting must match a numpy reference (AIQ-36/37)."""
    model = ts_testing.small_model("lenet5", num_classes=10)
    # Pin the DuckDB backend so this test actually exercises the GROUP BY
    # path (the new auto-select would pick host on a CPU box, gpu on CUDA).
    ts.build_efficiency_model(
        model, attach_to=["Conv2dNormActivation"], memory_limit="1GB"
    )
    model.eval()
    with torch.no_grad():
        model(torch.randn(8, 3, 64, 64))

    probes = ts.layers(model)
    assert probes
    for probe in probes.values():
        assert isinstance(probe, StateCaptureHook)
        counts = probe.counts()
        ids = probe.state_ids()
        # state_ids and counts are aligned over the unique-microstate set.
        assert len(ids) == len(counts)
        # Counts sum to the total in-window post-batch-dedup row count.
        assert int(counts.sum()) == probe.state_count
        # GROUP BY counts agree with a numpy reference over the raw rows.
        # raw_states is now unique-in-window, so source the reference from
        # the unaggregated Arrow table instead.
        s_col = probe.to_arrow().column("s").to_pylist()
        arr = np.stack([np.frombuffer(b, dtype=np.uint8) for b in s_col])
        _, ref = np.unique(arr, axis=0, return_counts=True)
        assert sorted(counts.tolist()) == sorted(ref.tolist())


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
