//! `_compress_tensor_pi8`: bit-pack a 2-D u8 array.

use ndarray::{Array2, ArrayView2};

use crate::arch::scalar;

/// Compress a 2-D u8 array into bit-packed u8 bytes. Non-zero values set
/// the corresponding bit.
pub fn compress(input: ArrayView2<'_, u8>) -> Array2<u8> {
    let rows = input.nrows();
    let cols = input.ncols();
    let out_cols = cols.div_ceil(8);
    let mut out = Array2::<u8>::zeros((rows, out_cols));

    for r in 0..rows {
        let row = input.row(r);
        let row_slice = row.as_slice().expect("row contiguous");
        let out_row = out.row_mut(r);
        let out_slice = out_row
            .into_slice()
            .expect("output row contiguous");

        for byte_idx in 0..out_cols {
            let start = byte_idx * 8;
            let end = (start + 8).min(cols);
            out_slice[byte_idx] = scalar::pack_byte_from_u8(&row_slice[start..end]);
        }
    }
    out
}
