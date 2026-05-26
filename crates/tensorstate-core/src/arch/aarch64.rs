//! aarch64 NEON paths. Mirror the AVX2 + BMI2 implementations on the
//! x86_64 side using NEON intrinsics. NEON is always available on
//! ARMv8+ (Apple Silicon, Graviton, modern Linux ARM) so no runtime
//! detection is needed — the cfg gate alone is sufficient.

use std::arch::aarch64::*;

/// Pack 8 contiguous f32 values into a u8 byte. Bit i = (vals[i] > 0).
///
/// NEON has no direct `movemask` equivalent for float comparison, so we
/// compute the lane masks, AND each lane with its bit position, and
/// horizontal-add to assemble the byte. Two Q registers cover 8 f32
/// (4 lanes each).
#[inline]
pub fn pack_byte_from_f32_neon(vals: &[f32]) -> u8 {
    debug_assert!(vals.len() == 8);
    // SAFETY: requires NEON, which is mandatory on ARMv8+ (the only
    // aarch64 baseline we target). Loads are unaligned via `vld1q_f32`.
    unsafe {
        let v0 = vld1q_f32(vals.as_ptr());
        let v1 = vld1q_f32(vals.as_ptr().add(4));
        let zero = vdupq_n_f32(0.0);
        // u32x4 lanes: 0xFFFFFFFF where vi > 0, else 0.
        let m0 = vcgtq_f32(v0, zero);
        let m1 = vcgtq_f32(v1, zero);
        // Bit-position masks: lane i carries 1<<i. Two halves cover bits
        // 0..3 and 4..7 of the output byte.
        let bits_lo: uint32x4_t = vld1q_u32([1u32, 2, 4, 8].as_ptr());
        let bits_hi: uint32x4_t = vld1q_u32([16u32, 32, 64, 128].as_ptr());
        let masked_lo = vandq_u32(m0, bits_lo);
        let masked_hi = vandq_u32(m1, bits_hi);
        // Horizontal add gives the OR over disjoint bits.
        (vaddvq_u32(masked_lo) | vaddvq_u32(masked_hi)) as u8
    }
}

/// Pack 8 contiguous u8 values into a u8 byte. Bit i = (vals[i] != 0).
///
/// Single D register (uint8x8_t) holds all 8 input lanes. Compare-not-zero
/// per lane, AND with bit-position mask, horizontal add.
#[inline]
pub fn pack_byte_from_u8_neon(vals: &[u8]) -> u8 {
    debug_assert!(vals.len() == 8);
    // SAFETY: ARMv8 NEON baseline; unaligned load.
    unsafe {
        let v = vld1_u8(vals.as_ptr());
        let zero = vdup_n_u8(0);
        // u8x8 lanes: 0xFF where zero, 0x00 where non-zero. Invert to
        // get 0xFF where non-zero.
        let neq = vmvn_u8(vceq_u8(v, zero));
        // Bit-position mask: lane i carries 1<<i.
        let bits: uint8x8_t = vld1_u8([1u8, 2, 4, 8, 16, 32, 64, 128].as_ptr());
        let masked = vand_u8(neq, bits);
        vaddv_u8(masked)
    }
}
