"""Byte-equality validation + CPU benchmark across Cython and Rust backends.

Validates that the new Rust CPU primitives produce bit-identical output to
the Cython AVX/BMI2 implementations, and benchmarks both at varying sizes.

Prerequisites:
- The Rust extension is built automatically by `uv sync`.
- The Cython extension must be built explicitly via
  ``bash scripts/build-cython.sh`` before running this script.

Usage::

    bash scripts/build-cython.sh
    uv run python benches/cython_vs_rust.py
"""

import time
import sys

import numpy as np

try:
    import TensorState._TensorState as cython_ts
except ImportError:
    print(
        "Cython extension not built. Run `bash scripts/build-cython.sh` first.",
        file=sys.stderr,
    )
    sys.exit(1)

import TensorState._TensorState_rs as rust_ts


def bench(fn, *args, n_warmup: int = 3, n_reps: int = 10) -> float:
    """Return the mean wall-clock time in milliseconds for one fn call."""
    for _ in range(n_warmup):
        fn(*args)
    t0 = time.perf_counter()
    for _ in range(n_reps):
        fn(*args)
    return (time.perf_counter() - t0) / n_reps * 1000.0


def validate_compress_ps() -> bool:
    """Validate _compress_tensor_ps produces byte-identical output."""
    rng = np.random.default_rng(42)
    ok = True
    for n_rows in [1, 10, 100]:
        for n_cols in [1, 7, 8, 9, 13, 64, 127, 1024]:
            a = (rng.random((n_rows, n_cols)) - 0.5).astype(np.float32)
            cy = cython_ts._compress_tensor_ps(a)
            rs = rust_ts._compress_tensor_ps(a)
            if not np.array_equal(cy, rs):
                print(f"  MISMATCH _compress_tensor_ps n_rows={n_rows} n_cols={n_cols}")
                ok = False
    return ok


def validate_compress_pi8() -> bool:
    """Validate _compress_tensor_pi8 produces byte-identical output."""
    rng = np.random.default_rng(43)
    ok = True
    for n_rows in [1, 10, 100]:
        for n_cols in [1, 7, 8, 9, 13, 64, 127, 1024]:
            a = rng.choice([0, 1], size=(n_rows, n_cols), p=[0.5, 0.5]).astype(np.uint8)
            cy = cython_ts._compress_tensor_pi8(a)
            rs = rust_ts._compress_tensor_pi8(a)
            if not np.array_equal(cy, rs):
                print(f"  MISMATCH _compress_tensor_pi8 n_rows={n_rows} n_cols={n_cols}")
                ok = False
    return ok


def validate_decompress() -> bool:
    """Validate _decompress_tensor produces byte-identical output."""
    rng = np.random.default_rng(44)
    ok = True
    for n_rows in [1, 10, 100]:
        for n_cols in [1, 7, 8, 9, 13, 64, 127, 1024]:
            a = (rng.random((n_rows, n_cols)) - 0.5).astype(np.float32)
            compressed = cython_ts._compress_tensor_ps(a)
            cy = cython_ts._decompress_tensor(compressed, n_cols)
            rs = rust_ts._decompress_tensor(compressed.flatten(), n_cols)
            if not np.array_equal(cy, rs):
                print(f"  MISMATCH _decompress_tensor n_rows={n_rows} n_cols={n_cols}")
                ok = False
    return ok


def validate_lex_sort() -> bool:
    """Validate _lex_sort produces equivalent ordering (sorted rows match)."""
    rng = np.random.default_rng(45)
    ok = True
    for state_count in [1, 10, 100, 1000]:
        for n_cols in [1, 4, 8, 16]:
            states = rng.integers(0, 256, size=(state_count, n_cols), dtype=np.uint8)
            cy_edges, cy_index = cython_ts._lex_sort(states, state_count)
            rs_edges, rs_index = rust_ts._lex_sort(states, state_count)
            if not np.array_equal(cy_edges, rs_edges):
                print(
                    f"  MISMATCH _lex_sort edges state_count={state_count} n_cols={n_cols}"
                )
                ok = False
            # Index ordering may differ for equal rows (both valid); the
            # sorted rows must match.
            if not np.array_equal(states[cy_index], states[rs_index]):
                print(
                    f"  MISMATCH _lex_sort sorted-rows state_count={state_count} n_cols={n_cols}"
                )
                ok = False
    return ok


def benchmark() -> None:
    """Print a Cython-vs-Rust performance comparison table."""
    sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    n_rows = 10000
    rng = np.random.default_rng(46)

    print(
        f"{'cols':>6}  {'Cython AVX':>14}  {'Rust scalar':>14}  {'Rust/Cython':>14}"
    )
    print("-" * 60)
    for n_cols in sizes:
        a = (rng.random((n_rows, n_cols)) - 0.5).astype(np.float32)
        t_cy = bench(cython_ts._compress_tensor_ps, a)
        t_rs = bench(rust_ts._compress_tensor_ps, a)
        ratio = t_rs / t_cy
        print(
            f"{n_cols:>6}  {t_cy:>10.2f} ms  {t_rs:>10.2f} ms  {ratio:>10.2f}x"
        )


def main() -> int:
    print("Byte-equality validation (Cython == Rust):")
    ok = validate_compress_ps() and validate_compress_pi8()
    ok = validate_decompress() and ok
    ok = validate_lex_sort() and ok
    print(f"  Overall: {'OK' if ok else 'FAIL'}")
    if not ok:
        return 1
    print()
    print("Benchmark (10000 rows; 3 warmup + 10 timed runs; mean):")
    benchmark()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
