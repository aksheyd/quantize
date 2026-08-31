//! Learned quantization helpers.

use pyo3::prelude::*;

use crate::error::length_mismatch;
use crate::input::{as_f32_values, as_i32_codes};
use crate::quantized::PyQuantized;

#[pyfunction]
pub fn refine<'py>(
    quantized: Bound<'py, PyQuantized>,
    values: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyQuantized>> {
    let owned = as_f32_values(&values)?;
    {
        let mut inner = quantized.borrow_mut();
        if inner.len() != owned.len() {
            return Err(length_mismatch(inner.len(), owned.len()));
        }
        if !owned.is_empty() {
            inner.refine(&owned);
        }
    }
    Ok(quantized)
}

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
