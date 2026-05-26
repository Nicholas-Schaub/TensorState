//! aarch64 NEON paths. Mirrors the Cython AVX2+BMI2 implementations using
//! NEON intrinsics. NEON is always available on ARMv8+ so no runtime
//! detection is needed.
//!
//! TODO: implement these. The scalar path is correct and sufficient for
//! now; NEON optimisations land incrementally.

#![allow(dead_code)]

/// Pack 8 contiguous f32 values into a u8 byte via NEON `vcgtq_f32` + a
/// manual shift+OR reduction (NEON has no direct movemask equivalent).
#[inline]
pub fn pack_byte_from_f32_neon(vals: &[f32]) -> u8 {
    crate::arch::scalar::pack_byte_from_f32(vals)
}

/// Pack 8 contiguous u8 values into a u8 byte via NEON shift+OR reduction.
#[inline]
pub fn pack_byte_from_u8_neon(vals: &[u8]) -> u8 {
    crate::arch::scalar::pack_byte_from_u8(vals)
}
