//! `_compress_tensor_ps`: bit-pack a 2-D f32 array.
//!
//! The public `compress` function dispatches at first call to the best
//! available implementation: AVX2 on x86_64 with AVX2 support, otherwise
//! the scalar fallback. The dispatch decision is cached for the lifetime
//! of the process via `OnceLock`, so per-call overhead is a single
//! pointer load.
//!
//! The `compress_scalar` loop is intentionally tight — it relies on
//! LLVM's auto-vectoriser to lower to vector compare + reduce on x86
//! and ARM. Do not pessimise this loop without re-benchmarking against
//! the SIMD path.

use std::sync::OnceLock;

use ndarray::{Array2, ArrayView2};

use crate::arch::scalar;

type CompressF32Fn = fn(ArrayView2<'_, f32>) -> Array2<u8>;

static DISPATCH: OnceLock<CompressF32Fn> = OnceLock::new();

/// Compress a 2-D f32 array into bit-packed u8 bytes (8 bits per byte along
/// the last axis). Values > 0 set the corresponding bit.
///
/// Output shape: `(rows, ceil(cols / 8))`. If `cols` is not a multiple of 8,
/// the final byte is padded with zero bits in the high-order positions.
pub fn compress(input: ArrayView2<'_, f32>) -> Array2<u8> {
    let f = DISPATCH.get_or_init(select);
    f(input)
}

fn select() -> CompressF32Fn {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return compress_avx2_safe;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return compress_neon;
    }
    compress_scalar
}

/// Identifier for the runtime-selected dispatch path. Returns one of
/// `"avx2"`, `"neon"`, or `"scalar"`. Trigger dispatch initialisation
/// if it has not happened yet by calling `compress` on a zero-row input.
pub fn dispatch_name() -> &'static str {
    let _ = DISPATCH.get_or_init(select);
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return "avx2";
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "neon";
    }
    "scalar"
}

/// Scalar reference implementation. Always available.
pub fn compress_scalar(input: ArrayView2<'_, f32>) -> Array2<u8> {
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

/// Safe wrapper around the AVX2 implementation. Only registered as the
/// dispatch target when `is_x86_feature_detected!("avx2")` returned true.
#[cfg(target_arch = "x86_64")]
fn compress_avx2_safe(input: ArrayView2<'_, f32>) -> Array2<u8> {
    // SAFETY: `select` only stores this fn pointer when AVX2 is available.
    unsafe { compress_avx2(input) }
}

/// NEON path on aarch64. Always available on ARMv8+ (Apple Silicon,
/// Graviton, modern ARM servers), so no runtime detection needed.
#[cfg(target_arch = "aarch64")]
fn compress_neon(input: ArrayView2<'_, f32>) -> Array2<u8> {
    use crate::arch::aarch64::pack_byte_from_f32_neon;

    let rows = input.nrows();
    let cols = input.ncols();
    let out_cols = cols.div_ceil(8);
    let mut out = Array2::<u8>::zeros((rows, out_cols));

    let full_bytes = cols / 8;
    let tail = cols % 8;

    for r in 0..rows {
        let row = input.row(r);
        let row_slice = row.as_slice().expect("row contiguous");
        let out_row = out.row_mut(r);
        let out_slice = out_row.into_slice().expect("output row contiguous");

        for byte_idx in 0..full_bytes {
            let start = byte_idx * 8;
            out_slice[byte_idx] =
                pack_byte_from_f32_neon(&row_slice[start..start + 8]);
        }
        if tail > 0 {
            let start = full_bytes * 8;
            out_slice[full_bytes] = scalar::pack_byte_from_f32(&row_slice[start..]);
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
