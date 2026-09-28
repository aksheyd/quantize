//! Scale storage width: `f32`, `f16`, or `bf16`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::error::QuantizeError;

/// Runtime choice of `Quantized<f32>`, `Quantized<f16>`, or `Quantized<bf16>`.
#[pyclass(
    eq,
    frozen,
    hash,
    skip_from_py_object,
    name = "Scale",
    module = "quantize"
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PyScale {
    F32,
    F16,
    Bf16,
}

impl PyScale {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "f32" => Some(Self::F32),
            "f16" => Some(Self::F16),
            "bf16" => Some(Self::Bf16),
            _ => None,
        }
    }
}

impl std::fmt::Display for PyScale {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Scale.{self:?}")
    }
}

/// A `scale` argument is a `Scale`, or its name, like `"f16"`. `str()` reads
/// the name, so the 0-d array that `numpy.load` returns for a saved name
/// works too.
impl FromPyObject<'_, '_> for PyScale {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        if let Ok(scale) = obj.cast::<PyScale>() {
            return Ok(*scale.get());
        }
        match Self::from_name(obj.str()?.to_str()?) {
            Some(scale) => Ok(scale),
            None => Err(PyErr::new::<QuantizeError, _>(format!(
                "scale must be a Scale or its name, 'f32', 'f16', or 'bf16', got {}",
                obj.repr()?
            ))),
        }
    }
}

#[pymethods]
impl PyScale {
    /// The name that `scale=` also accepts: `'f32'`, `'f16'`, or `'bf16'`.
    #[getter]
    fn name(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }

    fn __getstate__(&self) -> &'static str {
        self.name()
    }

    #[staticmethod]
    fn _from_pickle(name: &str) -> PyResult<Self> {
        Self::from_name(name).ok_or_else(|| PyValueError::new_err("malformed pickle state"))
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, (&'static str,))> {
        let callable = slf.as_any().getattr("_from_pickle")?;
        Ok((callable, (slf.get().name(),)))
    }
}
