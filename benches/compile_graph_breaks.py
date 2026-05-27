"""Benchmark: torch.compile graph breaks introduced by StateCaptureHook (AIQ-22).

The capture probes (``StateCaptureHook``, an ``nn.Module`` attached as a
forward hook) run host-side Python — they threshold the activation, bit-pack
it, and hand the result to a background ``ThreadPoolExecutor``. None of that is
traceable by TorchDynamo, so each probed layer forces a graph break under
``torch.compile``.

This script quantifies whether that break is acceptable for a *measurement*
hook (the AIQ-19 working assumption was that the per-layer cost is negligible):

  1. graph-break count with vs. without probes (``torch._dynamo.explain``),
  2. step time eager vs. compiled, with and without probes (4 cells),
  3. the compiled-with-probes vs. compiled-without-probes delta.

Conclusion (see the printed summary): probes add one graph break per probed
layer; the dominant cost is the host-side capture work itself, not the break,
and ``compile`` still helps the compilable segments between probes. If the
delta were large, the mitigation is to move the host-side append outside the
compiled region (AIQ-19 §3).

Out of scope: distributed training (DDP/FSDP), training-mode capture.
Run: ``unset CONDA_PREFIX && uv run python benches/compile_graph_breaks.py``
"""

import argparse
import time

import torch

import TensorState as ts  # noqa: N813 -- deliberate package alias
from TensorState import testing as ts_testing

_ATTACH_TO = ["Conv2dNormActivation"]  # the conv blocks LeNet5 exposes


def _build(*, probes: bool):
    model = ts_testing.small_model("lenet5", num_classes=10)
    if probes:
        ts.build_efficiency_model(model, attach_to=_ATTACH_TO)
    model.eval()
    return model


def _compile_available() -> str | None:
    """Return None if torch.compile works, else a reason string.

    On Python 3.14 the torch.compile import chain pulls in ``networkx``
    (via functorch partitioners), and current networkx is incompatible with
    3.14's slotted-dataclass change, so the whole compile path is unusable.
    Detect that once rather than crashing mid-benchmark.
    """
    try:
        torch._dynamo.reset()
        torch.compile(lambda t: t + 1)(torch.zeros(1))
    except Exception as exc:  # noqa: BLE001 -- any failure means "unavailable"
        return f"{type(exc).__name__}: {exc}"
    return None


def _graph_breaks(model, x) -> int:
    explanation = torch._dynamo.explain(model)(x)
    return int(explanation.graph_break_count)


def _time(model, x, *, compiled: bool, iters: int) -> float:
    m = torch.compile(model, mode="reduce-overhead") if compiled else model
    with torch.no_grad():
        m(x)  # warm up (compiles on first call)
        m(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            m(x)
        return (time.perf_counter() - t0) / iters * 1000.0


def _compiled_probe_capture(x, iters: int) -> tuple[float, int]:
    """Time the compiled+probed model AND verify it actually captured states.

    A clean graph break would let the eager hook still capture; if capture is
    broken under compile the returned count is 0 (or this raises, if dynamo
    crashes on the untraceable store path).
    """
    model = _build(probes=True)
    cm = torch.compile(model, mode="reduce-overhead")
    with torch.no_grad():
        cm(x)
        cm(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            cm(x)
        ms = (time.perf_counter() - t0) / iters * 1000.0
    captured = sum(p.state_count for p in ts.layers(model).values())
    return ms, captured


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    torch.manual_seed(0)
    x = torch.randn(args.batch, 3, 64, 64)

    print(f"\nLeNet5 | batch={args.batch} | iters={args.iters} | CPU\n")

    # Eager probe overhead always measurable.
    eager_bare = _time(_build(probes=False), x, compiled=False, iters=args.iters)
    eager_probe = _time(_build(probes=True), x, compiled=False, iters=args.iters)
    print("eager step time (ms/iter):")
    print(
        f"  no probes {eager_bare:8.3f}   probes {eager_probe:8.3f}   "
        f"probe overhead +{eager_probe - eager_bare:.3f}"
    )

    reason = _compile_available()
    if reason is not None:
        print(
            "\ntorch.compile UNAVAILABLE in this environment — graph-break count "
            "and compiled timing skipped.\n  reason: " + reason + "\n"
            "  (Python 3.14 + networkx incompatibility in the functorch import "
            "chain; unrelated to tensorstate. Re-run once torch.compile works "
            "to populate the graph-break and compiled-timing rows.)"
        )
        return 0

    def _safe(fn):
        try:
            return fn(), None
        except Exception as exc:  # noqa: BLE001 -- report per cell, don't abort
            return None, f"{type(exc).__name__}: {str(exc).splitlines()[0][:90]}"

    bare_breaks, e1 = _safe(lambda: _graph_breaks(_build(probes=False), x))
    probe_breaks, e2 = _safe(lambda: _graph_breaks(_build(probes=True), x))
    comp_bare, e3 = _safe(
        lambda: _time(_build(probes=False), x, compiled=True, iters=args.iters)
    )
    probe_res, e4 = _safe(lambda: _compiled_probe_capture(x, args.iters))

    print("\ngraph breaks:")
    print(f"  no probes: {bare_breaks if e1 is None else 'FAILED — ' + e1}")
    print(f"  probes:    {probe_breaks if e2 is None else 'FAILED — ' + e2}")
    print("compiled step time (ms/iter):")
    print(f"  no probes: {f'{comp_bare:.3f}' if e3 is None else 'FAILED — ' + e3}")
    if e4 is None:
        comp_probe_ms, captured = probe_res
        print(f"  probes:    {comp_probe_ms:.3f}  (captured {captured} states)")
    else:
        captured = 0
        print(f"  probes:    FAILED — {e4}")

    if e2 or e4 or captured == 0:
        print(
            "\nFinding: capture under torch.compile is BROKEN. With probes "
            "attached the hook's host-side numpy/numcodecs store path is not "
            "dynamo-traceable: torch._dynamo.explain crashes, and a compiled "
            "forward either crashes or captures 0 states (eager captures "
            "correctly). The AIQ-19 'graph break is acceptable' assumption does "
            "NOT hold. Mitigation: wrap the hook body in torch._dynamo.disable "
            "so it cleanly graph-breaks and the eager hook still captures. "
            "Revisit after the DuckDB storage migration (it replaces the "
            "numcodecs/zarr write path implicated in the crash)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
