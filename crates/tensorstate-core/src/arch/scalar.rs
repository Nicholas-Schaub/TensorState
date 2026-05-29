//! Scalar (portable, no SIMD) implementations of the bit-packing primitives.
//!
//! These are the always-available reference implementations. SIMD paths in
//! sibling modules must produce byte-identical output to these functions.

/// Pack 8 sequential f32 values into one u8 byte. Bit i = (vals[i] > 0).
#[inline]
pub fn pack_byte_from_f32(vals: &[f32]) -> u8 {
    debug_assert!(vals.len() <= 8);
    let mut byte: u8 = 0;
    for (i, &v) in vals.iter().enumerate() {
        if v > 0.0 {
            byte |= 1u8 << i;
        }
    }
    byte
}

/// Pack 8 sequential u8 values into one u8 byte. Bit i = (vals[i] != 0).
#[inline]
pub fn pack_byte_from_u8(vals: &[u8]) -> u8 {
    debug_assert!(vals.len() <= 8);
    let mut byte: u8 = 0;
    for (i, &v) in vals.iter().enumerate() {
        if v != 0 {
            byte |= 1u8 << i;
        }
    }
    byte
}

/// Unpack one u8 byte into 8 boolean values, bit 0 -> bools[0], bit 7 -> bools[7].
#[inline]
pub fn unpack_byte_to_bool(byte: u8, bools: &mut [bool]) {
    debug_assert!(bools.len() <= 8);
    for (i, b) in bools.iter_mut().enumerate() {
        *b = (byte >> i) & 1 == 1;
    }
}
