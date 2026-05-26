//! x86_64 SIMD paths. Mirror the Cython AVX2 + BMI2 implementations.
//!
//! Functions in this module require AVX2 or BMI2 respectively; the caller
//! must runtime-detect via `is_x86_feature_detected!` before calling them.
//! Calling without the required feature is undefined behavior.
//!
//! The dispatch wrappers in `compress_f32` and `compress_u8` perform the
//! feature detection at module-load time and cache the resulting function
//! pointer in an `OnceLock`.

use std::arch::x86_64::*;

/// Pack 8 contiguous f32 values into a u8 byte via AVX2's
/// `_mm256_cmp_ps` + `_mm256_movemask_ps`. Bit i = (vals[i] > 0).
///
/// SAFETY: caller must guarantee AVX2 is available and `vals.len() == 8`.
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn pack_byte_from_f32_avx2(vals: &[f32]) -> u8 {
    debug_assert!(vals.len() == 8);
    let v = _mm256_loadu_ps(vals.as_ptr());
    let zero = _mm256_setzero_ps();
    // _CMP_GT_OQ = 0x1e — ordered, non-signaling, greater-than.
    let cmp = _mm256_cmp_ps::<{ _CMP_GT_OQ }>(v, zero);
    let mask = _mm256_movemask_ps(cmp);
    mask as u8
}

/// Pack 8 contiguous u8 bool values into a u8 byte via BMI2's `_pext_u64`.
/// Bit i = (vals[i] != 0).
///
/// SAFETY: caller must guarantee BMI2 is available and `vals.len() == 8`.
/// `vals.as_ptr()` does not need 8-byte alignment.
#[target_feature(enable = "bmi2")]
#[inline]
pub unsafe fn pack_byte_from_u8_bmi2(vals: &[u8]) -> u8 {
    debug_assert!(vals.len() == 8);
    // Unaligned 8-byte load; treat any non-zero byte as a 1-bit by mapping
    // bytes to a {0,1}-valued u64 first.
    let raw = std::ptr::read_unaligned(vals.as_ptr() as *const u64);
    // (raw | (raw >> 4)) on its own keeps bytes non-zero iff any nibble was
    // non-zero; combined with the OR of low/high halves below produces a
    // single bit-per-byte representation cheap enough that the BMI2 PEXT
    // dominates the cost.
    let lo = raw | (raw >> 4);
    let lo = lo | (lo >> 2);
    let lo = lo | (lo >> 1);
    // Now bit 0 of each byte is set iff the original byte was non-zero.
    let extracted = _pext_u64(lo, 0x0101_0101_0101_0101);
    extracted as u8
}

/// Pack 16 contiguous f32 values into 2 u8 bytes via two AVX2 movemasks.
/// Output is little-endian: `out[0]` covers vals[0..8], `out[1]` covers
/// vals[8..16].
///
/// SAFETY: caller must guarantee AVX2 is available and `vals.len() == 16`.
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn pack_two_bytes_from_f32_avx2(vals: &[f32]) -> u16 {
    debug_assert!(vals.len() == 16);
    let v0 = _mm256_loadu_ps(vals.as_ptr());
    let v1 = _mm256_loadu_ps(vals.as_ptr().add(8));
    let zero = _mm256_setzero_ps();
    let m0 = _mm256_movemask_ps(_mm256_cmp_ps::<{ _CMP_GT_OQ }>(v0, zero)) as u32;
    let m1 = _mm256_movemask_ps(_mm256_cmp_ps::<{ _CMP_GT_OQ }>(v1, zero)) as u32;
    (m0 & 0xff) as u16 | (((m1 & 0xff) as u16) << 8)
}

/// Pack 32 contiguous u8 bool values into a u32 (= 4 packed bytes,
/// little-endian) via AVX2's `_mm256_cmpeq_epi8` + `_mm256_movemask_epi8`.
/// Bit i of the result is set iff `vals[i] != 0`. Treats every non-zero byte
/// as a 1-bit regardless of its specific value.
///
/// SAFETY: caller must guarantee AVX2 is available and `vals.len() == 32`.
/// The 32-byte load does not require alignment.
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn pack_four_bytes_from_u8_avx2(vals: &[u8]) -> u32 {
    debug_assert!(vals.len() == 32);
    let v = _mm256_loadu_si256(vals.as_ptr() as *const __m256i);
    let zero = _mm256_setzero_si256();
    // cmpeq_epi8 sets each byte to 0xFF where input == 0, else 0x00.
    let eq_zero = _mm256_cmpeq_epi8(v, zero);
    // movemask_epi8 extracts the high bit of each byte into a 32-bit mask.
    let mask = _mm256_movemask_epi8(eq_zero) as u32;
    // We have 1 bits where byte == 0; flip to get 1 bits where byte != 0.
    !mask
}
