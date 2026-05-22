//! x86_64 SIMD paths. Mirrors the Cython AVX2 + BMI2 implementations.
//!
//! Functions in this module require AVX2 or BMI2 respectively; the caller
//! must runtime-detect via `is_x86_feature_detected!` before calling them.
//! Calling without the required feature is undefined behavior.
//!
//! TODO(AIQ-6 follow-up): implement these. The scalar path is correct and
//! sufficient for the first commit; SIMD optimisations are tracked separately
//! and land incrementally.

#![allow(dead_code)]

/// Pack 8 contiguous f32 values into a u8 byte via AVX2's
/// `_mm256_cmp_ps` + `_mm256_movemask_ps`. Caller must guarantee AVX2 is
/// available.
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn pack_byte_from_f32_avx2(vals: &[f32]) -> u8 {
    // For now, fall back to scalar. A future commit will replace this with
    // _mm256_loadu_ps + _mm256_cmp_ps(_, zero, _CMP_GT_OQ) + _mm256_movemask_ps.
    crate::arch::scalar::pack_byte_from_f32(vals)
}

/// Pack 8 contiguous u8 bool values into a u8 byte via BMI2's `_pext_u64`.
/// Caller must guarantee BMI2 is available.
#[target_feature(enable = "bmi2")]
#[inline]
pub unsafe fn pack_byte_from_u8_bmi2(vals: &[u8]) -> u8 {
    // For now, fall back to scalar. A future commit will replace this with
    // _pext_u64(<u64-load of vals>, 0x0101010101010101).
    crate::arch::scalar::pack_byte_from_u8(vals)
}
