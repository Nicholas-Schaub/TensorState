//! SIMD/scalar byte-equivalence tests for the bit-pack paths.
//!
//! These verify that the AVX2 and BMI2 implementations produce
//! byte-identical output to the scalar reference across edge cases
//! and randomized inputs. Tests are gated on `cfg(target_arch =
//! "x86_64")` for the AVX2/BMI2 cases; the dispatch wrapper falls
//! back to scalar on other architectures so the equivalence is
//! trivially satisfied there.

use _tensorstate_rs::{compress_f32, compress_u8};
use ndarray::Array2;

fn f32_inputs() -> Vec<Array2<f32>> {
    let mut out = Vec::new();
    // Edge cases on cols dim: partial-byte tail, exact byte, pair-byte
    // boundary, all-zero, all-ones, alternating, mixed positive/negative.
    for &cols in &[1usize, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128] {
        // Mixed positive/negative
        let mut mixed = Array2::<f32>::zeros((4, cols));
        for r in 0..4 {
            for c in 0..cols {
                let v = ((r * 31 + c * 7) as i32 % 13 - 6) as f32;
                mixed[(r, c)] = v;
            }
        }
        out.push(mixed);

        // All-zero
        out.push(Array2::<f32>::zeros((3, cols)));

        // All-positive
        out.push(Array2::<f32>::from_elem((3, cols), 1.0));

        // Alternating sign per column
        let mut alt = Array2::<f32>::zeros((2, cols));
        for r in 0..2 {
            for c in 0..cols {
                alt[(r, c)] = if c % 2 == 0 { 1.0 } else { -1.0 };
            }
        }
        out.push(alt);
    }
    out
}

fn u8_inputs() -> Vec<Array2<u8>> {
    let mut out = Vec::new();
    for &cols in &[1usize, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128] {
        // Alternating non-zero / zero
        let mut alt = Array2::<u8>::zeros((4, cols));
        for r in 0..4 {
            for c in 0..cols {
                alt[(r, c)] = if (r + c) % 2 == 0 { 0 } else { 1 };
            }
        }
        out.push(alt);

        out.push(Array2::<u8>::zeros((3, cols)));
        out.push(Array2::<u8>::from_elem((3, cols), 1));
        out.push(Array2::<u8>::from_elem((3, cols), 255));

        // Random-ish pattern with non-{0,1} values
        let mut weird = Array2::<u8>::zeros((2, cols));
        for r in 0..2 {
            for c in 0..cols {
                weird[(r, c)] = ((r * 17 + c * 5) % 256) as u8;
            }
        }
        out.push(weird);
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_matches_scalar_f32() {
    if !std::arch::is_x86_feature_detected!("avx2") {
        eprintln!("skip: AVX2 not available on this CPU");
        return;
    }
    for input in f32_inputs() {
        let scalar = compress_f32::compress_scalar(input.view());
        // SAFETY: AVX2 checked above.
        let simd = unsafe { compress_f32::compress_avx2(input.view()) };
        assert_eq!(
            scalar, simd,
            "AVX2 disagreed with scalar on shape {:?}",
            input.dim()
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_matches_scalar_u8() {
    if !std::arch::is_x86_feature_detected!("avx2") {
        eprintln!("skip: AVX2 not available on this CPU");
        return;
    }
    for input in u8_inputs() {
        let scalar = compress_u8::compress_scalar(input.view());
        // SAFETY: AVX2 checked above.
        let simd = unsafe { compress_u8::compress_avx2(input.view()) };
        assert_eq!(
            scalar, simd,
            "AVX2 disagreed with scalar on shape {:?}",
            input.dim()
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn bmi2_matches_scalar_u8() {
    if !std::arch::is_x86_feature_detected!("bmi2") {
        eprintln!("skip: BMI2 not available on this CPU");
        return;
    }
    for input in u8_inputs() {
        let scalar = compress_u8::compress_scalar(input.view());
        // SAFETY: BMI2 checked above.
        let simd = unsafe { compress_u8::compress_bmi2(input.view()) };
        assert_eq!(
            scalar, simd,
            "BMI2 disagreed with scalar on shape {:?}",
            input.dim()
        );
    }
}

#[test]
fn dispatch_matches_scalar_f32() {
    for input in f32_inputs() {
        let scalar = compress_f32::compress_scalar(input.view());
        let dispatched = compress_f32::compress(input.view());
        assert_eq!(
            scalar,
            dispatched,
            "dispatched f32 path disagreed with scalar on shape {:?}",
            input.dim()
        );
    }
}

#[test]
fn dispatch_matches_scalar_u8() {
    for input in u8_inputs() {
        let scalar = compress_u8::compress_scalar(input.view());
        let dispatched = compress_u8::compress(input.view());
        assert_eq!(
            scalar,
            dispatched,
            "dispatched u8 path disagreed with scalar on shape {:?}",
            input.dim()
        );
    }
}
