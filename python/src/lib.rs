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
    InvalidBitsError, InvalidBlockError, InvalidToleranceError, LengthMismatchError,
    NotAMatrixError, QuantizeError, ScaleOutOfRangeError, ShapeMismatchError,
    ToleranceTooTightError,
};
use crate::input::as_f32_array;
use crate::learned::{alternate, fit_scale_and_zero_point, refine};
use crate::quantized::PyQuantized;
use crate::scale::PyScale;
use crate::scheme::PyScheme;

fn quantize_values(
    py: Python<'_>,
    values: Bound<'_, PyAny>,
    scale: PyScale,
    scheme: impl FnOnce(usize) -> Scheme,
) -> PyResult<PyQuantized> {
    let (array, shape) = as_f32_array(&values)?;
    let values = array.as_slice()?;
    let scheme = scheme(values.len());
    py.detach(|| PyQuantized::from_scheme(scheme, values, &shape, scale))
}

// Each `text_signature` repeats its `signature` so that `help()` shows the
// default scale as `'f32'`: a signature can only show plain values, so the
// generated one would show `scale=...`.

/// Quantize `values` symmetrically: each block of `block` values shares one
/// scale, and each value is stored as a signed `bits`-wide code that decodes
/// as `code * scale`. `bits` runs from 2 to 16. `scale` is how each scale is
/// stored: `Scale.F32`, `Scale.F16`, `Scale.BF16`, or its name. `values` is a
/// 1-D or 2-D array or a list, and a 2-D array keeps its shape.
#[pyfunction]
#[pyo3(
    name = "quantize",
    signature = (values, bits = 8, block = 32, *, scale = PyScale::F32),
    text_signature = "(values, bits=8, block=32, *, scale='f32')"
)]
fn symmetric_quantize(
    py: Python<'_>,
    values: Bound<'_, PyAny>,
    #[pyo3(from_py_with = input::bits_argument)] bits: u32,
    #[pyo3(from_py_with = input::block_argument)] block: usize,
    scale: PyScale,
) -> PyResult<PyQuantized> {
    quantize_values(py, values, scale, |_| Scheme::Symmetric { bits, block })
}

/// Quantize `values` like `quantize`, with one scale for the whole tensor.
#[pyfunction]
#[pyo3(
    signature = (values, bits = 8, *, scale = PyScale::F32),
    text_signature = "(values, bits=8, *, scale='f32')"
)]
fn quantize_tensor(
    py: Python<'_>,
    values: Bound<'_, PyAny>,
    #[pyo3(from_py_with = input::bits_argument)] bits: u32,
    scale: PyScale,
) -> PyResult<PyQuantized> {
    quantize_values(py, values, scale, |len| Scheme::Symmetric {
        bits,
        block: len.max(1),
    })
}

/// Quantize `values` asymmetrically: each block of `block` values gets a
/// scale and a zero-point, so values that aren't centered on zero can use
/// every code. Each value decodes as `(code - zero_point) * scale`. The other
/// arguments work as in `quantize`.
///
/// `Scale.F16` and `Scale.BF16` round each zero-point, which caps the
/// accuracy above about 10 bits, or sooner on blocks far from zero, so use
/// `Scale.F32` there.
#[pyfunction]
#[pyo3(
    signature = (values, bits = 8, block = 32, *, scale = PyScale::F32),
    text_signature = "(values, bits=8, block=32, *, scale='f32')"
)]
fn asymmetric_quantize(
    py: Python<'_>,
    values: Bound<'_, PyAny>,
    #[pyo3(from_py_with = input::bits_argument)] bits: u32,
    #[pyo3(from_py_with = input::block_argument)] block: usize,
    scale: PyScale,
) -> PyResult<PyQuantized> {
    quantize_values(py, values, scale, |_| Scheme::Asymmetric { bits, block })
}

/// Quantize `values` like `asymmetric.quantize`, with one scale and one
/// zero-point for the whole tensor.
#[pyfunction]
#[pyo3(
    signature = (values, bits = 8, *, scale = PyScale::F32),
    text_signature = "(values, bits=8, *, scale='f32')"
)]
fn asymmetric_quantize_tensor(
    py: Python<'_>,
    values: Bound<'_, PyAny>,
    #[pyo3(from_py_with = input::bits_argument)] bits: u32,
    scale: PyScale,
) -> PyResult<PyQuantized> {
    quantize_values(py, values, scale, |len| Scheme::Asymmetric {
        bits,
        block: len.max(1),
    })
}

/// Quantize `values` asymmetrically, giving each block of `block` values the
/// fewest bits, from 2 to 8, whose rounding error, half a step, is at most
/// `tolerance`. The other arguments work as in `quantize`.
///
/// `tolerance` is in the same units as the values, so one number can be loose
/// for one layer and tight for the next. Pick it from their spread, like
/// `tolerance=0.1 * np.std(values)`, which gives normal weights about 5 bits
/// a block, whatever their size.
///
/// If even 8 bits can't round a block within `tolerance`, this raises
/// `ToleranceTooTightError`, which gives the smallest tolerance that every
/// block meets. With `Scale.F16` or `Scale.BF16`, a value can land slightly
/// past the tolerance, and several times past on blocks far from zero, so use
/// `Scale.F32` there.
#[pyfunction]
#[pyo3(
    signature = (values, block = 32, *, tolerance, scale = PyScale::F32),
    text_signature = "(values, block=32, *, tolerance, scale='f32')"
)]
fn adaptive_quantize(
    py: Python<'_>,
    values: Bound<'_, PyAny>,
    #[pyo3(from_py_with = input::block_argument)] block: usize,
    tolerance: f32,
    scale: PyScale,
) -> PyResult<PyQuantized> {
    quantize_values(py, values, scale, |_| Scheme::Adaptive { block, tolerance })
}

#[pymodule]
#[pyo3(name = "_native")]
fn native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<QuantizeError>()?;
    m.add_class::<InvalidBitsError>()?;
    m.add_class::<InvalidBlockError>()?;
    m.add_class::<InvalidToleranceError>()?;
    m.add_class::<ToleranceTooTightError>()?;
    m.add_class::<ScaleOutOfRangeError>()?;
    m.add_class::<LengthMismatchError>()?;
    m.add_class::<ShapeMismatchError>()?;
    m.add_class::<NotAMatrixError>()?;
    m.add_class::<PyScale>()?;
    m.add_class::<PyScheme>()?;
    m.add_class::<PyQuantized>()?;
    m.add_function(wrap_pyfunction!(symmetric_quantize, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(asymmetric_quantize, m)?)?;
    m.add_function(wrap_pyfunction!(asymmetric_quantize_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(adaptive_quantize, m)?)?;
    m.add_function(wrap_pyfunction!(refine, m)?)?;
    m.add_function(wrap_pyfunction!(alternate, m)?)?;
    m.add_function(wrap_pyfunction!(fit_scale_and_zero_point, m)?)?;
    Ok(())
}
