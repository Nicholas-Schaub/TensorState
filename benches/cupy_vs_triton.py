"""GPU benchmark: native PyTorch vs Triton (current) vs CuPy (reference).

Validates that the Triton GPU kernel from AIQ-7 produces byte-identical
output to the Cython AVX baseline AND to a reference CuPy ElementwiseKernel
implementation, and compares their throughput on the same hardware.

Prerequisites:
- A CUDA-capable GPU.
- Build the Cython extension via ``bash scripts/build-cython.sh``.
- ``uv pip install cupy-cuda12x`` for the CuPy reference (not a required
  project dependency; CuPy is included only for benchmark comparison).

Originally written during AIQ-7 to demonstrate that the Triton kernel
matches the CuPy ElementwiseKernel; preserved in-tree for future
regression / parity checks.
"""

import time

import numpy as np
import torch

# Cython reference is the bit-pack gold standard.
import TensorState._TensorState as cython_ts
from TensorState import States  # noqa: E402

try:
    import cupy
except ImportError as exc:
    raise ImportError(
        "cupy-cuda12x is not installed. Install with `uv pip install "
        "cupy-cuda12x` to run this benchmark."
    ) from exc


# Inline CuPy ElementwiseKernel matching the original (pre-AIQ-7) impl.
_compress_kernel_cupy = cupy.ElementwiseKernel(
    "raw T myarray, raw int64 myarray_size, raw int64 in_cols, raw int64 out_cols, raw int64 stride",
    "uint8 packed",
    """
    long row = i / out_cols;
    long col = (i % out_cols) * stride;
    long k = row * in_cols + col;
    long nvals = (col + stride - 1 < in_cols) ? stride : in_cols - col;
    for (long j = 0; j < nvals; ++j) {
        int bit = myarray[k+j] != 0;
        packed |= bit << j;
    }""",
    "packbits_kernel",
)


def cupy_compress(a_gpu: cupy.ndarray) -> cupy.ndarray:
    myarray = (a_gpu > 0).ravel()
    nrows = a_gpu.shape[0]
    ncols = (a_gpu.shape[1] + 7) // 8
    packed = cupy.zeros((nrows * ncols,), dtype=cupy.uint8)
    stride = min([8, a_gpu.shape[1]])
    return _compress_kernel_cupy(
        myarray, myarray.size, a_gpu.shape[1], ncols, stride, packed
    ).reshape(nrows, ncols)


def run_size(n_rows: int, n_cols: int, n_warmup: int = 3, n_reps: int = 10):
    rng = np.random.default_rng(42)
    a_np = (rng.random((n_rows, n_cols)) - 0.5).astype(np.float32)
    a_torch_cpu = torch.from_numpy(a_np)
    a_torch_cuda = a_torch_cpu.cuda()
    a_cupy = cupy.asarray(a_np)

    # Byte-equality validation: every backend must match the Cython AVX.
    out_cython = cython_ts._compress_tensor_ps(a_np)
    out_torch_cpu = States.compress_states(a_torch_cpu).cpu().numpy()
    torch.cuda.synchronize()
    out_torch_cuda = States.compress_states(a_torch_cuda).cpu().numpy()
    torch.cuda.synchronize()
    out_cupy = cupy.asnumpy(cupy_compress(a_cupy))
    cupy.cuda.runtime.deviceSynchronize()

    valid_torch_cpu = np.array_equal(out_cython, out_torch_cpu)
    valid_torch_cuda = np.array_equal(out_cython, out_torch_cuda)
    valid_cupy = np.array_equal(out_cython, out_cupy)

    def bench(fn, sync=None):
        for _ in range(n_warmup):
            fn()
        if sync:
            sync()
        t0 = time.perf_counter()
        for _ in range(n_reps):
            fn()
            if sync:
                sync()
        return (time.perf_counter() - t0) / n_reps

    t_cython = bench(lambda: cython_ts._compress_tensor_ps(a_np))
    t_torch_cpu = bench(lambda: States.compress_states(a_torch_cpu))
    t_torch_cuda = bench(
        lambda: States.compress_states(a_torch_cuda),
        sync=torch.cuda.synchronize,
    )
    t_cupy = bench(lambda: cupy_compress(a_cupy), sync=cupy.cuda.runtime.deviceSynchronize)

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "valid_torch_cpu": valid_torch_cpu,
        "valid_torch_cuda": valid_torch_cuda,
        "valid_cupy": valid_cupy,
        "t_cython_ms": t_cython * 1000,
        "t_torch_cpu_ms": t_torch_cpu * 1000,
        "t_torch_cuda_ms": t_torch_cuda * 1000,
        "t_cupy_ms": t_cupy * 1000,
    }


def main() -> int:
    sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    n_rows = 10000

    print("=" * 110)
    print(
        f"{'cols':>6}  {'Cython AVX':>14}  {'torch CPU':>12}  "
        f"{'Triton CUDA':>12}  {'CuPy CUDA':>12}  {'== Cython?':>20}"
    )
    print("-" * 110)

    for n_cols in sizes:
        r = run_size(n_rows, n_cols)
        bytes_match = (
            f"tCPU={'OK' if r['valid_torch_cpu'] else 'DIFF'}, "
            f"tCUDA={'OK' if r['valid_torch_cuda'] else 'DIFF'}, "
            f"cupy={'OK' if r['valid_cupy'] else 'DIFF'}"
        )
        print(
            f"{n_cols:>6}  {r['t_cython_ms']:>10.2f} ms  "
            f"{r['t_torch_cpu_ms']:>8.2f} ms  "
            f"{r['t_torch_cuda_ms']:>8.2f} ms  "
            f"{r['t_cupy_ms']:>8.2f} ms  {bytes_match}"
        )

    print("=" * 110)
    print(f"All sizes tested with {n_rows} rows; 3 warmup + 10 timed runs; mean reported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
