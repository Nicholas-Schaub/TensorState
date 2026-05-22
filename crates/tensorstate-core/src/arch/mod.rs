//! Architecture detection and SIMD dispatch.
//!
//! On x86_64 we runtime-detect AVX2 and BMI2 to pick between SIMD and scalar
//! paths. On aarch64 NEON is always available on ARMv8+ so we compile it in
//! unconditionally. On other architectures we fall back to scalar.

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

pub mod scalar;

/// Returns a human-readable description of the SIMD path selected at startup.
pub fn describe() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        let avx2 = std::arch::is_x86_feature_detected!("avx2");
        let bmi2 = std::arch::is_x86_feature_detected!("bmi2");
        match (avx2, bmi2) {
            (true, true) => "x86_64 AVX2+BMI2".to_string(),
            (true, false) => "x86_64 AVX2 (no BMI2)".to_string(),
            (false, true) => "x86_64 BMI2 (no AVX2)".to_string(),
            (false, false) => "x86_64 scalar (no AVX2/BMI2)".to_string(),
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        "aarch64 NEON".to_string()
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        format!("{} scalar", std::env::consts::ARCH)
    }
}
