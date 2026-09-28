//! Learned quantization helpers.

use pyo3::prelude::*;

use crate::error::{from_quantize, length_mismatch};
use crate::input::{as_f32_array, as_f32_values, as_i32_codes};
use crate::quantized::PyQuantized;

/// Refit each block's scale, and its zero-point if it has one, to lower the
/// error against `values`, the numbers `quantized` was quantized from. The
/// codes don't move, so the tensor keeps its size.
///
/// This changes `quantized` in place and returns it. Call
/// `quantized.copy()` first to keep the original.
#[pyfunction]
pub fn refine<'py>(
    quantized: Bound<'py, PyQuantized>,
    values: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyQuantized>> {
    let (owned, _) = as_f32_array(&values)?;
    quantized
        .borrow_mut()
        .refine(&owned)
        .map_err(from_quantize)?;
    Ok(quantized)
}

/// Refit like `refine`, then round each of `values` to its nearest code on
/// its block's new line, and repeat until no code moves, for at most 100
/// passes. The error ends no higher than `refine` alone leaves it, and the
/// tensor keeps its kind, bit widths, and size.
///
/// Like `refine`, this changes `quantized` in place and returns it. Call
/// `quantized.copy()` first to keep the original.
#[pyfunction]
pub fn alternate<'py>(
    quantized: Bound<'py, PyQuantized>,
    values: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyQuantized>> {
    let (owned, _) = as_f32_array(&values)?;
    quantized
        .borrow_mut()
        .alternate(&owned)
        .map_err(from_quantize)?;
    Ok(quantized)
}

/// The `(scale, zero_point)` that best fit
/// `values ≈ scale * (codes - zero_point)`, by least squares. `codes` holds
/// one integer code per value, like `q.unpacked_codes`.
#[pyfunction]
pub fn fit_scale_and_zero_point(
    values: Bound<'_, PyAny>,
    codes: Bound<'_, PyAny>,
) -> PyResult<(f32, f32)> {
    let owned_values = as_f32_values(&values)?;
    let owned_codes = as_i32_codes(&codes)?;
    if owned_values.len() != owned_codes.len() {
        return Err(length_mismatch(owned_values.len(), owned_codes.len()));
    }
    Ok(quantize::learned::fit_scale_and_zero_point(
        &owned_values,
        &owned_codes,
    ))
}
