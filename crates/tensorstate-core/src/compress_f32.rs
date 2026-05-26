//! `_compress_tensor_ps`: bit-pack a 2-D f32 array.

use ndarray::{Array2, ArrayView2};

use crate::arch::scalar;

/// Compress a 2-D f32 array into bit-packed u8 bytes (8 bits per byte along
/// the last axis). Values > 0 set the corresponding bit.
///
/// Output shape: `(rows, ceil(cols / 8))`. If `cols` is not a multiple of 8,
/// the final byte is padded with zero bits in the high-order positions.
pub fn compress(input: ArrayView2<'_, f32>) -> Array2<u8> {
    let rows = input.nrows();
    let cols = input.ncols();
    let out_cols = cols.div_ceil(8);
    let mut out = Array2::<u8>::zeros((rows, out_cols));

    for r in 0..rows {
        let row = input.row(r);
        let row_slice = row.as_slice().expect("row contiguous");
        let out_row = out.row_mut(r);
        let out_slice = out_row.into_slice().expect("output row contiguous");

        for byte_idx in 0..out_cols {
            let start = byte_idx * 8;
            let end = (start + 8).min(cols);
            out_slice[byte_idx] = scalar::pack_byte_from_f32(&row_slice[start..end]);
        }
    }
    out
}

/// SIMD version using hand-tuned AVX2 intrinsics for the full-byte chunks.
/// Falls back to scalar for the partial-byte tail.
///
/// SAFETY: caller must guarantee AVX2 is available
/// (`is_x86_feature_detected!("avx2")`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn compress_avx2(input: ArrayView2<'_, f32>) -> Array2<u8> {
    use crate::arch::x86_64::{pack_byte_from_f32_avx2, pack_two_bytes_from_f32_avx2};

    let rows = input.nrows();
    let cols = input.ncols();
    let out_cols = cols.div_ceil(8);
    let mut out = Array2::<u8>::zeros((rows, out_cols));

    let full_bytes = cols / 8;
    let tail = cols % 8;
    let pair_bytes = full_bytes / 2;
    let single_after_pairs = full_bytes % 2;

    for r in 0..rows {
        let row = input.row(r);
        let row_slice = row.as_slice().expect("row contiguous");
        let out_row = out.row_mut(r);
        let out_slice = out_row.into_slice().expect("output row contiguous");

        // Process two bytes (16 f32) per iteration when possible.
        for pair_idx in 0..pair_bytes {
            let start = pair_idx * 16;
            let packed = unsafe {
                pack_two_bytes_from_f32_avx2(&row_slice[start..start + 16])
            };
            out_slice[pair_idx * 2] = (packed & 0xff) as u8;
            out_slice[pair_idx * 2 + 1] = ((packed >> 8) & 0xff) as u8;
        }
        // Handle a trailing single full byte.
        if single_after_pairs == 1 {
            let start = pair_bytes * 16;
            out_slice[pair_bytes * 2] = unsafe {
                pack_byte_from_f32_avx2(&row_slice[start..start + 8])
            };
        }
        // Scalar for the < 8 element tail.
        if tail > 0 {
            let start = full_bytes * 8;
            out_slice[full_bytes] = scalar::pack_byte_from_f32(&row_slice[start..]);
        }
    }
    out
}
