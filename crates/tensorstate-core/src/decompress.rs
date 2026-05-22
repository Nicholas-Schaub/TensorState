//! `_decompress_tensor`: unpack a 1-D u8 buffer back into a 2-D boolean array.

use ndarray::Array2;
use numpy::ndarray::ArrayView1;

use crate::arch::scalar;

/// Decompress a flat bit-packed buffer into `(n_rows, n_neurons)` booleans.
///
/// `input` is expected to be the row-major bit-packed buffer of shape
/// `(n_rows * bytes_per_row)` where `bytes_per_row = ceil(n_neurons / 8)`.
pub fn decompress(input: ArrayView1<'_, u8>, n_neurons: usize) -> Array2<bool> {
    let bytes_per_row = n_neurons.div_ceil(8);
    let n_rows = input.len() / bytes_per_row;
    let mut out = Array2::<bool>::default((n_rows, n_neurons));

    let input_slice = input.as_slice().expect("input contiguous");

    for r in 0..n_rows {
        let row_start = r * bytes_per_row;
        let mut out_row = out.row_mut(r);
        let out_slice = out_row
            .as_slice_mut()
            .expect("output row contiguous");

        for byte_idx in 0..bytes_per_row {
            let byte = input_slice[row_start + byte_idx];
            let bit_start = byte_idx * 8;
            let bit_end = (bit_start + 8).min(n_neurons);
            let chunk = &mut out_slice[bit_start..bit_end];
            scalar::unpack_byte_to_bool(byte, chunk);
        }
    }
    out
}
