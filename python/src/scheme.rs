//! Runtime quantization schemes.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::error::from_quantize;
use crate::input::{bits_argument, block_argument};
use crate::quantize_values;
use crate::quantized::PyQuantized;
use crate::scale::PyScale;

/// A quantization method and its settings, picked at run time.
/// `Scheme.symmetric`, `Scheme.asymmetric`, and `Scheme.adaptive` build one,
/// and `quantize` runs it. `Scheme.Q8_32` and `Scheme.Q4_32` are symmetric
/// 8-bit and 4-bit codes, with blocks of 32.
///
/// `Scheme(text)` reads a scheme written like a call to one of those
/// methods, without the `Scheme.`, such as
/// `Scheme("adaptive(block=32, tolerance=0.002)")`, or a constant's name,
/// such as `Scheme("Q4_32")`. Text that isn't a scheme, or that holds a
/// value `quantize` would reject, like `bits=99`, raises `QuantizeError`.
///
/// Pickles and copies load through `Scheme(text)`, so a scheme that
/// `quantize` would reject, like `Scheme.symmetric(bits=99)`, raises the
/// same error when it's unpickled or copied. `torch.load` accepts pickles
/// once `torch.serialization.add_safe_globals([Scheme])` allows the class.
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
    #[new]
    fn new(text: &str) -> PyResult<Self> {
        let inner = text.parse().map_err(from_quantize)?;
        Ok(Self { inner })
    }

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
    #[pyo3(signature = (block = 32, *, tolerance))]
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

    /// The largest rounding error an adaptive scheme allows for any value, in
    /// the values' own units, or `None`.
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

    /// Pickles saved by quantize-py 0.3.0 and earlier call this with the
    /// parts that `__getstate__` returns.
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

    // Pickles call the class with the scheme's text, so that `torch.load`
    // loads them once `add_safe_globals([Scheme])` allows it, as with
    // `Quantized`.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> (Bound<'py, PyType>, (String,)) {
        (slf.get_type(), (slf.get().inner.to_string(),))
    }
}
