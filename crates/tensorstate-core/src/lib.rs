//! TensorState core: Rust port of the Cython CPU primitives.
//!
//! Four primitives are exposed via PyO3:
//!
//! - `compress_tensor_ps(input: array<f32, 2>) -> array<u8, 2>` — bit-pack a
//!   2-D float32 array by thresholding `> 0` and packing 8 bits per byte
//!   along the last axis.
//! - `compress_tensor_pi8(input: array<u8, 2>) -> array<u8, 2>` — bit-pack a
//!   2-D uint8 / bool array (any non-zero value treated as 1).
//! - `decompress_tensor(input: array<u8, 2>, num_neurons: usize) -> array<bool, 2>` —
//!   unpack a bit-packed array back to a boolean array of `(rows, num_neurons)`.
//! - `lex_sort(states: array<u8, 2>, state_count: usize) -> (bin_edges, index)` —
//!   radix lex-sort the rows so identical rows are adjacent, returning the
//!   sort index and the bin edges between distinct unique rows.
//!
//! SIMD strategy: each primitive has a scalar reference implementation that
//! is always available. Architecture-specific SIMD lives in modules under
//! [`arch`] and is selected by runtime feature detection on x86_64 (AVX2 +
//! BMI2) and compile-time selection on aarch64 (NEON, always available on
//! ARMv8+). The scalar path remains the fallback.

use pyo3::prelude::*;

mod arch;
mod compress_f32;
mod compress_u8;
mod decompress;
mod lex_sort;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

/// Bit-pack a 2-D float32 array. Values > 0 become firing bits.
#[pyfunction]
#[pyo3(name = "_compress_tensor_ps")]
fn py_compress_tensor_ps<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let view = input.as_array();
    let out = compress_f32::compress(view);
    Ok(out.into_pyarray(py))
}

/// Bit-pack a 2-D uint8 array. Non-zero values become firing bits.
#[pyfunction]
#[pyo3(name = "_compress_tensor_pi8")]
fn py_compress_tensor_pi8<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, u8>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let view = input.as_array();
    let out = compress_u8::compress(view);
    Ok(out.into_pyarray(py))
}

/// Hand-tuned AVX2 version of `_compress_tensor_ps`. Requires AVX2.
#[pyfunction]
#[pyo3(name = "_compress_tensor_ps_simd")]
fn py_compress_tensor_ps_simd<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    #[cfg(target_arch = "x86_64")]
    {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "AVX2 not available on this CPU; use _compress_tensor_ps",
            ));
        }
        let view = input.as_array();
        let out = unsafe { compress_f32::compress_avx2(view) };
        Ok(out.into_pyarray(py))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (py, input);
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "_compress_tensor_ps_simd is x86_64-only",
        ))
    }
}

/// Hand-tuned BMI2 version of `_compress_tensor_pi8`. Requires BMI2.
/// Processes 8 input bytes -> 1 output byte per intrinsic call.
#[pyfunction]
#[pyo3(name = "_compress_tensor_pi8_bmi2")]
fn py_compress_tensor_pi8_bmi2<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, u8>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    #[cfg(target_arch = "x86_64")]
    {
        if !std::arch::is_x86_feature_detected!("bmi2") {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "BMI2 not available on this CPU; use _compress_tensor_pi8",
            ));
        }
        let view = input.as_array();
        let out = unsafe { compress_u8::compress_bmi2(view) };
        Ok(out.into_pyarray(py))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (py, input);
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "_compress_tensor_pi8_bmi2 is x86_64-only",
        ))
    }
}

/// Hand-tuned AVX2 version of `_compress_tensor_pi8`. Requires AVX2.
/// Processes 32 input bytes -> 4 output bytes per intrinsic call (4x the
/// throughput of the BMI2 path on CPUs where both are available).
#[pyfunction]
#[pyo3(name = "_compress_tensor_pi8_avx2")]
fn py_compress_tensor_pi8_avx2<'py>(
    py: Python<'py>,
    input: PyReadonlyArray2<'py, u8>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    #[cfg(target_arch = "x86_64")]
    {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "AVX2 not available on this CPU; use _compress_tensor_pi8",
            ));
        }
        let view = input.as_array();
        let out = unsafe { compress_u8::compress_avx2(view) };
        Ok(out.into_pyarray(py))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (py, input);
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "_compress_tensor_pi8_avx2 is x86_64-only",
        ))
    }
}

/// Decompress a bit-packed uint8 array back to a boolean array.
#[pyfunction]
#[pyo3(name = "_decompress_tensor")]
fn py_decompress_tensor<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'py, u8>,
    n_neurons: i64,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    if n_neurons <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "n_neurons must be positive",
        ));
    }
    let view = input.as_array();
    let out = decompress::decompress(view, n_neurons as usize);
    Ok(out.into_pyarray(py))
}

/// Radix lex-sort the rows of a uint8 array. Returns `(bin_edges, index)`.
#[pyfunction]
#[pyo3(name = "_lex_sort")]
fn py_lex_sort<'py>(
    py: Python<'py>,
    states: PyReadonlyArray2<'py, u8>,
    state_count: i64,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    if state_count < 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "state_count must be non-negative",
        ));
    }
    let view = states.as_array();
    let (bin_edges, index) = lex_sort::lex_sort(view, state_count as usize);
    Ok((bin_edges.into_pyarray(py), index.into_pyarray(py)))
}

/// Build info exposed to Python for diagnostics.
#[pyfunction]
fn _build_info() -> PyResult<String> {
    let arch_info = arch::describe();
    Ok(format!(
        "tensorstate-core {} (rustc {}, {})",
        env!("CARGO_PKG_VERSION"),
        env!("RUSTC_VERSION_INFO").trim(),
        arch_info,
    ))
}

#[pymodule]
#[pyo3(name = "_TensorState_rs")]
fn _tensorstate_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_compress_tensor_ps, m)?)?;
    m.add_function(wrap_pyfunction!(py_compress_tensor_pi8, m)?)?;
    m.add_function(wrap_pyfunction!(py_compress_tensor_ps_simd, m)?)?;
    m.add_function(wrap_pyfunction!(py_compress_tensor_pi8_bmi2, m)?)?;
    m.add_function(wrap_pyfunction!(py_compress_tensor_pi8_avx2, m)?)?;
    m.add_function(wrap_pyfunction!(py_decompress_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(py_lex_sort, m)?)?;
    m.add_function(wrap_pyfunction!(_build_info, m)?)?;
    Ok(())
}
