//! Scale storage width: `f32`, `f16`, or `bf16`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Runtime choice of `Quantized<f32>`, `Quantized<f16>`, or `Quantized<bf16>`.
#[pyclass(eq, frozen, hash, from_py_object, name = "Scale", module = "quantize")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PyScale {
    F32,
    F16,
    Bf16,
}

impl PyScale {
    fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }

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

#[pymethods]
impl PyScale {
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
