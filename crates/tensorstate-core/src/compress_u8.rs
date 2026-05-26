//! `_compress_tensor_pi8`: bit-pack a 2-D u8 array.
//!
//! IMPORTANT — output type contract: the output is always `Array2<u8>` and
//! the row stride is `ceil(cols / 8)` bytes. This is load-bearing for memory
//! efficiency on odd-shape layers (e.g., N=9 neurons costs 2 bytes/row at
//! u8 granularity vs 4 bytes/row at u32 or 8 bytes/row at u64). Do not
//! "optimize" the output type to a wider integer; SIMD paths must accumulate
//! the mask in a wider register internally but split it to uint8 bytes when
//! writing to the output array.
//!
//! The public `compress` function dispatches at first call to the best
//! available implementation: AVX2 > BMI2 > scalar on x86_64, NEON on
//! aarch64. Dispatch decision is cached via `OnceLock`.
//!
//! The `compress_scalar` loop relies on LLVM auto-vectorisation for the
//! fallback case. Do not pessimise it without re-benchmarking against
//! the SIMD path.

use std::sync::OnceLock;

use ndarray::{Array2, ArrayView2};

use crate::arch::scalar;

type CompressU8Fn = fn(ArrayView2<'_, u8>) -> Array2<u8>;

static DISPATCH: OnceLock<CompressU8Fn> = OnceLock::new();

/// Compress a 2-D u8 array into bit-packed u8 bytes. Non-zero values set
/// the corresponding bit.
pub fn compress(input: ArrayView2<'_, u8>) -> Array2<u8> {
    let f = DISPATCH.get_or_init(select);
    f(input)
}

fn select() -> CompressU8Fn {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return compress_avx2_safe;
        }
        if std::arch::is_x86_feature_detected!("bmi2") {
            return compress_bmi2_safe;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return compress_neon;
    }
    compress_scalar
}

/// Identifier for the runtime-selected dispatch path. Returns one of
/// `"avx2"`, `"bmi2"`, `"neon"`, or `"scalar"`.
pub fn dispatch_name() -> &'static str {
    let _ = DISPATCH.get_or_init(select);
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return "avx2";
        }
        if std::arch::is_x86_feature_detected!("bmi2") {
            return "bmi2";
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "neon";
    }
    "scalar"
}

/// Scalar reference implementation. Always available.
pub fn compress_scalar(input: ArrayView2<'_, u8>) -> Array2<u8> {
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
            out_slice[byte_idx] = scalar::pack_byte_from_u8(&row_slice[start..end]);
        }
    }
    out
}

#[cfg(target_arch = "x86_64")]
fn compress_avx2_safe(input: ArrayView2<'_, u8>) -> Array2<u8> {
    // SAFETY: `select` only stores this fn pointer when AVX2 is available.
    unsafe { compress_avx2(input) }
}

#[cfg(target_arch = "x86_64")]
fn compress_bmi2_safe(input: ArrayView2<'_, u8>) -> Array2<u8> {
    // SAFETY: `select` only stores this fn pointer when BMI2 is available.
    unsafe { compress_bmi2(input) }
}

/// NEON path on aarch64. Always available on ARMv8+.
#[cfg(target_arch = "aarch64")]
fn compress_neon(input: ArrayView2<'_, u8>) -> Array2<u8> {
    use crate::arch::aarch64::pack_byte_from_u8_neon;

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
                pack_byte_from_u8_neon(&row_slice[start..start + 8]);
        }
        if tail > 0 {
            let start = full_bytes * 8;
            out_slice[full_bytes] = scalar::pack_byte_from_u8(&row_slice[start..]);
        }
    }
    out
}

/// SIMD version using hand-tuned BMI2 `_pext_u64` for full-byte chunks
/// (8 bytes per call). Falls back to scalar for the partial-byte tail.
///
/// SAFETY: caller must guarantee BMI2 is available
/// (`is_x86_feature_detected!("bmi2")`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
pub unsafe fn compress_bmi2(input: ArrayView2<'_, u8>) -> Array2<u8> {
    use crate::arch::x86_64::pack_byte_from_u8_bmi2;

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
            out_slice[byte_idx] = unsafe {
                pack_byte_from_u8_bmi2(&row_slice[start..start + 8])
            };
        }
        if tail > 0 {
            let start = full_bytes * 8;
            out_slice[full_bytes] = scalar::pack_byte_from_u8(&row_slice[start..]);
        }
    }
    out
}

/// SIMD version using hand-tuned AVX2 `cmpeq_epi8` + `movemask_epi8` for
/// 32-byte chunks (4 output bytes per call). Falls back to scalar for the
/// 1–3 remaining full bytes and the partial-byte tail.
///
/// SAFETY: caller must guarantee AVX2 is available
/// (`is_x86_feature_detected!("avx2")`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn compress_avx2(input: ArrayView2<'_, u8>) -> Array2<u8> {
    use crate::arch::x86_64::pack_four_bytes_from_u8_avx2;

    let rows = input.nrows();
    let cols = input.ncols();
    let out_cols = cols.div_ceil(8);
    let mut out = Array2::<u8>::zeros((rows, out_cols));

    let full_bytes = cols / 8;
    let tail = cols % 8;
    let quad_bytes = full_bytes / 4;
    let bytes_after_quads = full_bytes % 4;

    for r in 0..rows {
        let row = input.row(r);
        let row_slice = row.as_slice().expect("row contiguous");
        let out_row = out.row_mut(r);
        let out_slice = out_row.into_slice().expect("output row contiguous");

        // Process 32 input bytes -> 4 output bytes per iteration.
        for quad_idx in 0..quad_bytes {
            let start = quad_idx * 32;
            let packed = unsafe {
                pack_four_bytes_from_u8_avx2(&row_slice[start..start + 32])
            };
            // Split the u32 mask into 4 uint8 bytes to preserve memory
            // granularity (see module-level docstring).
            let out_base = quad_idx * 4;
            out_slice[out_base] = (packed & 0xff) as u8;
            out_slice[out_base + 1] = ((packed >> 8) & 0xff) as u8;
            out_slice[out_base + 2] = ((packed >> 16) & 0xff) as u8;
            out_slice[out_base + 3] = ((packed >> 24) & 0xff) as u8;
        }
        // 1-3 remaining full bytes: scalar handles them cheaply.
        for byte_off in 0..bytes_after_quads {
            let byte_idx = quad_bytes * 4 + byte_off;
            let start = byte_idx * 8;
            out_slice[byte_idx] =
                scalar::pack_byte_from_u8(&row_slice[start..start + 8]);
        }
        // Partial-byte tail (cols not multiple of 8).
        if tail > 0 {
            let start = full_bytes * 8;
            out_slice[full_bytes] = scalar::pack_byte_from_u8(&row_slice[start..]);
        }
    }
    out
}
