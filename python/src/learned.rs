//! Learned quantization helpers.

use pyo3::prelude::*;

use crate::error::{from_quantize, length_mismatch};
use crate::input::{as_f32_array, as_f32_values, as_i32_codes, check_shape};
use crate::quantized::PyQuantized;

// `refine` and `alternate` refit a snapshot of the tensor with the GIL
// released, then store it. `Arc::make_mut` copies the shared tensor before
// the refit changes it, so until it's stored, other threads read the tensor
// as it was.

/// Refit each block's scale, and its zero-point if it has one, to lower the
/// mean squared error against `values`, the numbers `quantized` was
/// quantized from, in the same shape. The codes don't move, so the tensor
/// keeps its size.
///
/// A block's worst error can still rise, so a value in an adaptive tensor can
/// land past the tolerance it was quantized with.
///
/// This changes `quantized` in place and returns it. Call
/// `quantized.copy()` first to keep the original.
///
/// Other threads keep running while it refits, and see `quantized` as it was
/// until it returns. If two threads refit the same tensor at once, both start
/// from the tensor as it was, and it keeps the result of whichever finishes
/// last.
#[pyfunction]
pub fn refine<'py>(
    quantized: Bound<'py, PyQuantized>,
    values: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyQuantized>> {
    let (array, values_shape) = as_f32_array(&values)?;
    let mut refined = quantized.get().snapshot();
    check_shape(values.py(), "values", &refined.shape(), &values_shape)?;
    let values = array.as_slice()?;
    quantized
        .py()
        .detach(|| refined.refine(values))
        .map_err(from_quantize)?;
    quantized.get().store(refined);
    Ok(quantized)
}

/// Refit like `refine`, then round each of `values` to its nearest code on
/// its block's new line, and repeat until no code moves, for at most 100
/// passes. The mean squared error ends no higher than `refine` alone leaves
/// it, and the tensor keeps its kind, bit widths, and size.
///
/// A block's worst error can still rise, so a value in an adaptive tensor can
/// land past the tolerance it was quantized with.
///
/// Like `refine`, this changes `quantized` in place, so call
/// `quantized.copy()` first to keep the original, and other threads keep
/// running while it refits, as `refine` describes. Unlike `refine`, it
/// returns whether the codes settled: `True` once no code moves, or `False`
/// if it stopped after 100 passes. Call it again while it returns `False`.
#[pyfunction]
pub fn alternate(quantized: Bound<'_, PyQuantized>, values: Bound<'_, PyAny>) -> PyResult<bool> {
    let (array, values_shape) = as_f32_array(&values)?;
    let mut alternated = quantized.get().snapshot();
    check_shape(values.py(), "values", &alternated.shape(), &values_shape)?;
    let values = array.as_slice()?;
    let settled = quantized
        .py()
        .detach(|| alternated.alternate(values))
        .map_err(from_quantize)?;
    quantized.get().store(alternated);
    Ok(settled)
}

/// The `(scale, zero_point)` that best fit
/// `values ≈ scale * (codes - zero_point)`, by least squares. `codes` holds
/// one integer code per value, like `q.unpacked_codes`.
#[pyfunction]
pub fn fit_scale_and_zero_point(
    values: Bound<'_, PyAny>,
    codes: Bound<'_, PyAny>,
) -> PyResult<(f32, f32)> {
    let array = as_f32_values(&values)?;
    let owned_codes = as_i32_codes(&codes)?;
    let values = array.as_slice()?;
    if values.len() != owned_codes.len() {
        return Err(length_mismatch("codes", values.len(), owned_codes.len()));
    }
    Ok(quantize::learned::fit_scale_and_zero_point(
        values,
        &owned_codes,
    ))
}
