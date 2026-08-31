//! Python exceptions.

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::PyClassInitializer;

#[pyclass(
    frozen,
    extends = PyException,
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

#[pyclass(frozen, extends = QuantizeError, name = "InvalidBitsError", module = "quantize")]
pub struct InvalidBitsError {
    #[pyo3(get)]
    bits: u32,
}

#[pymethods]
impl InvalidBitsError {
    #[new]
    fn new(bits: u32) -> PyClassInitializer<Self> {
        let message = format!("bit width {bits} is outside the supported range 2..=16");
        PyClassInitializer::from(QuantizeError::new(message)).add_subclass(Self { bits })
    }

    fn __str__(&self) -> String {
        format!(
            "bit width {} is outside the supported range 2..=16",
            self.bits
        )
    }
}

#[pyclass(frozen, extends = QuantizeError, name = "InvalidBlockError", module = "quantize")]
pub struct InvalidBlockError {
    #[pyo3(get)]
    block: usize,
}

#[pymethods]
impl InvalidBlockError {
    #[new]
    fn new(block: usize) -> PyClassInitializer<Self> {
        let message = format!("block size {block} must be at least 1");
        PyClassInitializer::from(QuantizeError::new(message)).add_subclass(Self { block })
    }

    fn __str__(&self) -> String {
        format!("block size {} must be at least 1", self.block)
    }
}

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
    fn new(expected: usize, got: usize) -> PyClassInitializer<Self> {
        let message = format!("length mismatch: expected {expected}, got {got}");
        PyClassInitializer::from(QuantizeError::new(message)).add_subclass(Self { expected, got })
    }

    fn __str__(&self) -> String {
        format!(
            "length mismatch: expected {}, got {}",
            self.expected, self.got
        )
    }
}

pub fn length_mismatch(expected: usize, got: usize) -> PyErr {
    PyErr::new::<LengthMismatchError, _>((expected, got))
}

pub fn from_quantize(err: quantize::Error) -> PyErr {
    match err {
        quantize::Error::InvalidBits { bits } => PyErr::new::<InvalidBitsError, _>(bits),
        quantize::Error::InvalidBlock { block } => PyErr::new::<InvalidBlockError, _>(block),
        quantize::Error::InvalidTolerance => PyErr::new::<InvalidToleranceError, _>(()),
        quantize::Error::LengthMismatch { expected, got } => {
            PyErr::new::<LengthMismatchError, _>((expected, got))
        }
    }
}
