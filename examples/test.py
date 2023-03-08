import time

import cupy
import numpy as np

import TensorState as ts

n_neurons = 128
num_threads = 1

# modified from cupy source
# https://github.com/cupy/cupy/blob/v8.1.0/cupy/_binary/packing.py#L16
_compress_kernel = cupy.core.ElementwiseKernel(
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


# modified from cupy source
# https://github.com/cupy/cupy/blob/v8.1.0/cupy/_binary/packing.py#L16
def _compress_states_cuda(states):
    myarray = (states > 0).ravel()
    nrows = states.shape[0]
    ncols = (states.shape[1] + 7) // 8
    packed_size = nrows * ncols
    packed = cupy.zeros((packed_size,), dtype=cupy.uint8)
    stride = min([8, states.shape[1]])
    return _compress_kernel(
        myarray, myarray.size, states.shape[1], ncols, stride, packed
    ).reshape(nrows, ncols)


# Generate some random numbers to act as state information
a = (np.random.rand(100000, n_neurons) - 0.5).astype(np.float32)
# a[a<=0] = 0.0

# b = ts.compress_states(a)
# print(b)
# bins,ind = ts.sort_states(b,10)
# print(b[ind])
# print(bins)
# print(ts.compress_states(a))
# print(np.packbits(a>0,axis=1,bitorder='little'))
# aa = cupy.asarray(a)
# print(str(aa.device))
# print(ts.compress_states(aa>0))

# Run state compression, np.float32 -> np.uint8
replicates = 500
aa = cupy.asarray(a)
start = time.time()
for _ in range(replicates):
    b = ts.compress_states(aa.get())
print(time.time() - start)

aa = cupy.asarray(a)
start = time.time()
for _ in range(replicates):
    b = ts.compress_states((aa > 0).get())
print(time.time() - start)

start = time.time()
for _ in range(replicates):
    b = np.packbits(a > 0, axis=1, bitorder="little")
print(time.time() - start)

aa = cupy.asarray(a)
start = time.time()
for _ in range(replicates):
    bb = _compress_states_cuda(aa > 0).get()
print(time.time() - start)

# Make sure cupy and numpy are equal
print(f"CuPy and NumPy equal: {np.array_equal(b, bb.get())}")

# Run and test decompression, np.uint8 -> np.bool_
c = ts.decompress_states(b, n_neurons)
print(c.dtype)
print(f"Decompressed equals original: {np.array_equal(a > 0, c)}")

# Run and test recompression, np.bool_ -> np.uint8
d = ts.compress_states(c)
print(d.dtype)
print(f"Recompressed equals original: {np.array_equal(b, d)}")
