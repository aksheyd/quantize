//! Runtime quantization schemes.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::quantize_values;
use crate::quantized::PyQuantized;
use crate::scale::PyScale;

#[pyclass(frozen, name = "Scheme", module = "quantize", eq, skip_from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub struct PyScheme {
    inner: quantize::Scheme,
}

type SchemePickle = (&'static str, Option<u32>, usize, Option<f32>);

impl PyScheme {
    fn pickle_parts(self) -> SchemePickle {
        match self.inner {
            quantize::Scheme::Symmetric { bits, block } => ("symmetric", Some(bits), block, None),
            quantize::Scheme::Asymmetric { bits, block } => ("asymmetric", Some(bits), block, None),
            quantize::Scheme::Adaptive { block, tolerance } => {
                ("adaptive", None, block, Some(tolerance))
            }
        }
    }
}

#[pymethods]
impl PyScheme {
    #[classattr]
    #[pyo3(name = "Q8_32")]
    fn q8_32() -> Self {
        Self {
            inner: quantize::Scheme::Q8_32,
        }
    }

    #[classattr]
    #[pyo3(name = "Q4_32")]
    fn q4_32() -> Self {
        Self {
            inner: quantize::Scheme::Q4_32,
        }
    }

    #[classmethod]
    #[pyo3(signature = (bits = 8, block = 32))]
    fn symmetric(_cls: &Bound<'_, PyType>, bits: u32, block: usize) -> Self {
        Self {
            inner: quantize::Scheme::Symmetric { bits, block },
        }
    }

    #[classmethod]
    #[pyo3(signature = (bits = 8, block = 32))]
    fn asymmetric(_cls: &Bound<'_, PyType>, bits: u32, block: usize) -> Self {
        Self {
            inner: quantize::Scheme::Asymmetric { bits, block },
        }
    }

    #[classmethod]
    #[pyo3(signature = (block = 32, tolerance = 0.001))]
    fn adaptive(_cls: &Bound<'_, PyType>, block: usize, tolerance: f32) -> Self {
        Self {
            inner: quantize::Scheme::Adaptive { block, tolerance },
        }
    }

    #[pyo3(signature = (values, *, scale = PyScale::F32))]
    fn quantize(
        &self,
        py: Python<'_>,
        values: Bound<'_, PyAny>,
        scale: PyScale,
    ) -> PyResult<PyQuantized> {
        quantize_values(py, values, scale, |_| self.inner)
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            quantize::Scheme::Symmetric { .. } => "symmetric",
            quantize::Scheme::Asymmetric { .. } => "asymmetric",
            quantize::Scheme::Adaptive { .. } => "adaptive",
        }
    }

    #[getter]
    fn bits(&self) -> Option<u32> {
        match self.inner {
            quantize::Scheme::Symmetric { bits, .. }
            | quantize::Scheme::Asymmetric { bits, .. } => Some(bits),
            quantize::Scheme::Adaptive { .. } => None,
        }
    }

    #[getter]
    fn block(&self) -> usize {
        match self.inner {
            quantize::Scheme::Symmetric { block, .. }
            | quantize::Scheme::Asymmetric { block, .. }
            | quantize::Scheme::Adaptive { block, .. } => block,
        }
    }

    #[getter]
    fn tolerance(&self) -> Option<f32> {
        match self.inner {
            quantize::Scheme::Adaptive { tolerance, .. } => Some(tolerance),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            quantize::Scheme::Symmetric { bits, block } => {
                format!("Scheme(kind='symmetric', bits={bits}, block={block})")
            }
            quantize::Scheme::Asymmetric { bits, block } => {
                format!("Scheme(kind='asymmetric', bits={bits}, block={block})")
            }
            quantize::Scheme::Adaptive { block, tolerance } => {
                format!("Scheme(kind='adaptive', block={block}, tolerance={tolerance})")
            }
        }
    }

    fn __hash__(&self) -> PyResult<isize> {
        Err(PyTypeError::new_err("unhashable type: 'Scheme'"))
    }

    fn __getstate__(&self) -> SchemePickle {
        self.pickle_parts()
    }

    #[staticmethod]
    fn _from_pickle(
        kind: &str,
        bits: Option<u32>,
        block: usize,
        tolerance: Option<f32>,
    ) -> PyResult<Self> {
        match (kind, bits, tolerance) {
            ("symmetric", Some(bits), None) => Ok(Self {
                inner: quantize::Scheme::Symmetric { bits, block },
            }),
            ("asymmetric", Some(bits), None) => Ok(Self {
                inner: quantize::Scheme::Asymmetric { bits, block },
            }),
            ("adaptive", None, Some(tolerance)) => Ok(Self {
                inner: quantize::Scheme::Adaptive { block, tolerance },
            }),
            _ => Err(PyValueError::new_err("malformed pickle state")),
        }
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, SchemePickle)> {
        let callable = slf.getattr("_from_pickle")?;
        Ok((callable, slf.get().pickle_parts()))
    }
}
