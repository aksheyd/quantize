//! Python exceptions.

use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;
use pyo3::PyClassInitializer;

/// The base class of the errors for bad arguments and data. It's a
/// `ValueError`, so `except ValueError` catches these errors too.
#[pyclass(
    frozen,
    extends = PyValueError,
    subclass,
    name = "QuantizeError",
    module = "quantize"
)]
pub struct QuantizeError {
    message: String,
}

#[pymethods]
impl QuantizeError {
    #[new]
    fn new(message: String) -> Self {
        Self { message }
    }

    fn __str__(&self) -> String {
        self.message.clone()
    }
}

/// `bits` is outside 2 to 16. `bits` holds the width that was asked for.
#[pyclass(frozen, extends = QuantizeError, name = "InvalidBitsError", module = "quantize")]
pub struct InvalidBitsError {
    #[pyo3(get)]
    bits: i64,
}

#[pymethods]
impl InvalidBitsError {
    #[new]
    fn new(bits: i64) -> PyClassInitializer<Self> {
        let message = format!("bits must be from 2 to 16, got {bits}");
        PyClassInitializer::from(QuantizeError::new(message)).add_subclass(Self { bits })
    }

    fn __str__(&self) -> String {
        format!("bits must be from 2 to 16, got {}", self.bits)
    }
}

/// `block` is less than 1. `block` holds the size that was asked for.
#[pyclass(frozen, extends = QuantizeError, name = "InvalidBlockError", module = "quantize")]
pub struct InvalidBlockError {
    #[pyo3(get)]
    block: i64,
}

#[pymethods]
impl InvalidBlockError {
    #[new]
    fn new(block: i64) -> PyClassInitializer<Self> {
        let message = format!("block must be at least 1, got {block}");
        PyClassInitializer::from(QuantizeError::new(message)).add_subclass(Self { block })
    }

    fn __str__(&self) -> String {
        format!("block must be at least 1, got {}", self.block)
    }
}

/// `tolerance` isn't a finite number greater than 0.
#[pyclass(
    frozen,
    extends = QuantizeError,
    name = "InvalidToleranceError",
    module = "quantize"
)]
pub struct InvalidToleranceError {}

#[pymethods]
impl InvalidToleranceError {
    #[new]
    fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(QuantizeError::new(
            "tolerance must be a finite number greater than 0".to_string(),
        ))
        .add_subclass(Self {})
    }

    fn __str__(&self) -> &'static str {
        "tolerance must be a finite number greater than 0"
    }
}

/// A block's scale or zero-point doesn't fit in its scale type, such as a
/// zero-point beyond f16's 65,504. `block_index` and `scale_type` say which.
#[pyclass(
    frozen,
    extends = QuantizeError,
    name = "ScaleOutOfRangeError",
    module = "quantize"
)]
pub struct ScaleOutOfRangeError {
    #[pyo3(get)]
    block_index: usize,
    #[pyo3(get)]
    scale_type: String,
}

#[pymethods]
impl ScaleOutOfRangeError {
    #[new]
    fn new(block_index: usize, scale_type: String) -> PyClassInitializer<Self> {
        let message = format!(
            "block {block_index}'s scale or zero-point doesn't fit in {scale_type}; use f32 scales"
        );
        PyClassInitializer::from(QuantizeError::new(message)).add_subclass(Self {
            block_index,
            scale_type,
        })
    }

    fn __str__(&self) -> String {
        format!(
            "block {}'s scale or zero-point doesn't fit in {}; use f32 scales",
            self.block_index, self.scale_type
        )
    }
}

/// An array holds `got` numbers where the tensor needs `expected`.
#[pyclass(
    frozen,
    extends = QuantizeError,
    name = "LengthMismatchError",
    module = "quantize"
)]
pub struct LengthMismatchError {
    #[pyo3(get)]
    expected: usize,
    #[pyo3(get)]
    got: usize,
}

#[pymethods]
impl LengthMismatchError {
    #[new]
    #[pyo3(signature = (expected, got, message = None))]
    fn new(expected: usize, got: usize, message: Option<String>) -> PyClassInitializer<Self> {
        let message =
            message.unwrap_or_else(|| format!("length mismatch: expected {expected}, got {got}"));
        PyClassInitializer::from(QuantizeError::new(message)).add_subclass(Self { expected, got })
    }
}

/// `len` values can't be split into rows of `columns` columns.
#[pyclass(
    frozen,
    extends = QuantizeError,
    name = "ShapeMismatchError",
    module = "quantize"
)]
pub struct ShapeMismatchError {
    #[pyo3(get)]
    len: usize,
    #[pyo3(get)]
    columns: usize,
}

#[pymethods]
impl ShapeMismatchError {
    #[new]
    fn new(len: usize, columns: usize) -> PyClassInitializer<Self> {
        let message = format!("{len} values can't be split into rows of {columns} columns");
        PyClassInitializer::from(QuantizeError::new(message)).add_subclass(Self { len, columns })
    }

    fn __str__(&self) -> String {
        format!(
            "{} values can't be split into rows of {} columns",
            self.len, self.columns
        )
    }
}

/// `matmul` needs a tensor quantized from a 2-D array, but this one holds a
/// flat vector of `len` values.
#[pyclass(frozen, extends = QuantizeError, name = "NotAMatrixError", module = "quantize")]
pub struct NotAMatrixError {
    #[pyo3(get)]
    len: usize,
}

#[pymethods]
impl NotAMatrixError {
    #[new]
    fn new(len: usize) -> PyClassInitializer<Self> {
        let message =
            format!("matmul needs a matrix, but this tensor is a flat vector of {len} values");
        PyClassInitializer::from(QuantizeError::new(message)).add_subclass(Self { len })
    }

    fn __str__(&self) -> String {
        format!(
            "matmul needs a matrix, but this tensor is a flat vector of {} values",
            self.len
        )
    }
}

/// A `LengthMismatchError` that names the argument, like "out must have
/// length 8, got 3".
pub fn length_mismatch(argument: &str, expected: usize, got: usize) -> PyErr {
    let message = format!("{argument} must have length {expected}, got {got}");
    PyErr::new::<LengthMismatchError, _>((expected, got, message))
}

pub fn from_quantize(err: quantize::Error) -> PyErr {
    match err {
        quantize::Error::InvalidBits { bits } => PyErr::new::<InvalidBitsError, _>(i64::from(bits)),
        quantize::Error::InvalidBlock { block } => PyErr::new::<InvalidBlockError, _>(block as i64),
        quantize::Error::InvalidTolerance => PyErr::new::<InvalidToleranceError, _>(()),
        quantize::Error::ScaleOutOfRange {
            block_index,
            scale_type,
        } => PyErr::new::<ScaleOutOfRangeError, _>((block_index, scale_type)),
        quantize::Error::LengthMismatch { expected, got } => {
            PyErr::new::<LengthMismatchError, _>((expected, got))
        }
        quantize::Error::ShapeMismatch { len, columns } => {
            PyErr::new::<ShapeMismatchError, _>((len, columns))
        }
        quantize::Error::NotAMatrix { len } => PyErr::new::<NotAMatrixError, _>(len),
        quantize::Error::OutputTooLarge { .. } => PyMemoryError::new_err(err.to_string()),
        quantize::Error::Malformed { .. } => PyValueError::new_err(err.to_string()),
    }
}
