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

/// Returns a human-readable description of the SIMD path actually
/// selected by the runtime dispatch in `compress_f32` and `compress_u8`.
///
/// The format is `"<arch> <f32-path>/<u8-path>"` — for example
/// `"x86_64 avx2/avx2"` on a Haswell+ Intel/AMD CPU, or
/// `"aarch64 neon/neon"` on Apple Silicon. On exotic configurations
/// (AVX2 present but BMI2 missing, or pre-Haswell x86) the two slashes
/// can differ, e.g. `"x86_64 scalar/bmi2"`.
pub fn describe() -> String {
    let arch = if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else {
        std::env::consts::ARCH
    };
    format!(
        "{arch} {}/{}",
        crate::compress_f32::dispatch_name(),
        crate::compress_u8::dispatch_name(),
    )
}
