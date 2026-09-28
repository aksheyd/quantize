//! Runtime quantization schemes.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::input::{bits_argument, block_argument};
use crate::quantize_values;
use crate::quantized::PyQuantized;
use crate::scale::PyScale;

/// A quantization method and its settings, picked at run time.
/// `Scheme.symmetric`, `Scheme.asymmetric`, and `Scheme.adaptive` build one,
/// and `quantize` runs it. `Scheme.Q8_32` and `Scheme.Q4_32` are symmetric
/// 8-bit and 4-bit codes, with blocks of 32.
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

    /// The method of `quantize`.
    #[classmethod]
    #[pyo3(signature = (bits = 8, block = 32))]
    fn symmetric(
        _cls: &Bound<'_, PyType>,
        #[pyo3(from_py_with = bits_argument)] bits: u32,
        #[pyo3(from_py_with = block_argument)] block: usize,
    ) -> Self {
        Self {
            inner: quantize::Scheme::Symmetric { bits, block },
        }
    }

    /// The method of `asymmetric.quantize`.
    #[classmethod]
    #[pyo3(signature = (bits = 8, block = 32))]
    fn asymmetric(
        _cls: &Bound<'_, PyType>,
        #[pyo3(from_py_with = bits_argument)] bits: u32,
        #[pyo3(from_py_with = block_argument)] block: usize,
    ) -> Self {
        Self {
            inner: quantize::Scheme::Asymmetric { bits, block },
        }
    }

    /// The method of `adaptive.quantize`.
    #[classmethod]
    #[pyo3(signature = (block = 32, tolerance = 0.001))]
    fn adaptive(
        _cls: &Bound<'_, PyType>,
        #[pyo3(from_py_with = block_argument)] block: usize,
        tolerance: f32,
    ) -> Self {
        Self {
            inner: quantize::Scheme::Adaptive { block, tolerance },
        }
    }

    /// Quantize `values` with this scheme. `scale` works as in `quantize`.
    #[pyo3(
        signature = (values, *, scale = PyScale::F32),
        text_signature = "($self, values, *, scale='f32')"
    )]
    fn quantize(
        &self,
        py: Python<'_>,
        values: Bound<'_, PyAny>,
        scale: PyScale,
    ) -> PyResult<PyQuantized> {
        quantize_values(py, values, scale, |_| self.inner)
    }

    /// `'symmetric'`, `'asymmetric'`, or `'adaptive'`.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            quantize::Scheme::Symmetric { .. } => "symmetric",
            quantize::Scheme::Asymmetric { .. } => "asymmetric",
            quantize::Scheme::Adaptive { .. } => "adaptive",
        }
    }

    /// The code width, or `None` for an adaptive scheme.
    #[getter]
    fn bits(&self) -> Option<u32> {
        match self.inner {
            quantize::Scheme::Symmetric { bits, .. }
            | quantize::Scheme::Asymmetric { bits, .. } => Some(bits),
            quantize::Scheme::Adaptive { .. } => None,
        }
    }

    /// How many values share each scale.
    #[getter]
    fn block(&self) -> usize {
        match self.inner {
            quantize::Scheme::Symmetric { block, .. }
            | quantize::Scheme::Asymmetric { block, .. }
            | quantize::Scheme::Adaptive { block, .. } => block,
        }
    }

    /// The rounding error an adaptive scheme aims for, or `None`.
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

    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

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
