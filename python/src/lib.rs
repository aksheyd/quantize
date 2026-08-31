//! Native Python bindings.

mod error;
mod input;
mod learned;
mod quantized;
mod scale;
mod scheme;

use ::quantize::Scheme;
use pyo3::prelude::*;

use crate::error::{
    InvalidBitsError, InvalidBlockError, InvalidToleranceError, LengthMismatchError, QuantizeError,
};
use crate::input::as_f32_values;
use crate::learned::{fit_scale_and_zero_point, refine};
use crate::quantized::PyQuantized;
use crate::scale::PyScale;
use crate::scheme::PyScheme;

fn quantize_values(
    py: Python<'_>,
    values: Bound<'_, PyAny>,
    scale: PyScale,
    scheme: impl FnOnce(usize) -> Scheme,
) -> PyResult<PyQuantized> {
    let values = as_f32_values(&values)?;
    let scheme = scheme(values.len());
    py.detach(|| PyQuantized::from_scheme(scheme, &values, scale))
}

#[pyfunction]
#[pyo3(name = "quantize", signature = (values, bits = 8, block = 32, *, scale = PyScale::F32))]
fn symmetric_quantize(
    py: Python<'_>,
    values: Bound<'_, PyAny>,
    bits: u32,
    block: usize,
    scale: PyScale,
) -> PyResult<PyQuantized> {
    quantize_values(py, values, scale, |_| Scheme::Symmetric { bits, block })
}

#[pyfunction]
#[pyo3(signature = (values, bits = 8, *, scale = PyScale::F32))]
fn quantize_tensor(
    py: Python<'_>,
    values: Bound<'_, PyAny>,
    bits: u32,
    scale: PyScale,
) -> PyResult<PyQuantized> {
    quantize_values(py, values, scale, |len| Scheme::Symmetric {
        bits,
        block: len.max(1),
    })
}

#[pyfunction]
#[pyo3(signature = (values, bits = 8, block = 32, *, scale = PyScale::F32))]
fn asymmetric_quantize(
    py: Python<'_>,
    values: Bound<'_, PyAny>,
    bits: u32,
    block: usize,
    scale: PyScale,
) -> PyResult<PyQuantized> {
    quantize_values(py, values, scale, |_| Scheme::Asymmetric { bits, block })
}

#[pyfunction]
#[pyo3(signature = (values, bits = 8, *, scale = PyScale::F32))]
fn asymmetric_quantize_tensor(
    py: Python<'_>,
    values: Bound<'_, PyAny>,
    bits: u32,
    scale: PyScale,
) -> PyResult<PyQuantized> {
    quantize_values(py, values, scale, |len| Scheme::Asymmetric {
        bits,
        block: len.max(1),
    })
}

#[pyfunction]
#[pyo3(signature = (values, block = 32, tolerance = 0.001, *, scale = PyScale::F32))]
fn adaptive_quantize(
    py: Python<'_>,
    values: Bound<'_, PyAny>,
    block: usize,
    tolerance: f32,
    scale: PyScale,
) -> PyResult<PyQuantized> {
    quantize_values(py, values, scale, |_| Scheme::Adaptive { block, tolerance })
}

#[pymodule]
#[pyo3(name = "_native")]
fn native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.2.1")?;
    m.add_class::<QuantizeError>()?;
    m.add_class::<InvalidBitsError>()?;
    m.add_class::<InvalidBlockError>()?;
    m.add_class::<InvalidToleranceError>()?;
    m.add_class::<LengthMismatchError>()?;
    m.add_class::<PyScale>()?;
    m.add_class::<PyScheme>()?;
    m.add_class::<PyQuantized>()?;
    m.add_function(wrap_pyfunction!(symmetric_quantize, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(asymmetric_quantize, m)?)?;
    m.add_function(wrap_pyfunction!(asymmetric_quantize_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(adaptive_quantize, m)?)?;
    m.add_function(wrap_pyfunction!(refine, m)?)?;
    m.add_function(wrap_pyfunction!(fit_scale_and_zero_point, m)?)?;
    Ok(())
}
