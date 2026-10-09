//! Scale storage width: `f32`, `f16`, or `bf16`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::error::QuantizeError;

/// How each block's scale and zero-point are stored. `Scale.F32` keeps them
/// exactly, in 4 bytes each. `Scale.F16` and `Scale.BF16` round them to 2
/// bytes: f16 keeps more digits, and bf16 more range. Rounded zero-points cap
/// the accuracy of asymmetric codes above about 10 bits, so use `Scale.F32`
/// there. `scale=` also takes the name that `name` returns, and
/// `Scale(name)` gives the scale back, like `Scale("f16")`. Pickles load
/// through it, so `torch.load` accepts them once
/// `torch.serialization.add_safe_globals([Scale])` allows the class.
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
    #[pyo3(name = "BF16")]
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
        write!(f, "Scale.{}", self.name().to_uppercase())
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
    #[new]
    fn new(name: PyScale) -> Self {
        name
    }

    /// The name that `scale=` also accepts: `'f32'`, `'f16'`, or `'bf16'`.
    #[getter]
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }

    fn __getstate__(&self) -> &'static str {
        self.name()
    }

    /// Pickles saved by quantize-py 0.3.0 and earlier call this with the name.
    #[staticmethod]
    fn _from_pickle(name: &str) -> PyResult<Self> {
        Self::from_name(name).ok_or_else(|| PyValueError::new_err("malformed pickle state"))
    }

    // Pickles call the class with the name, so that `torch.load` loads them
    // once `add_safe_globals([Scale])` allows it, as with `Quantized`.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> (Bound<'py, PyType>, (&'static str,)) {
        (slf.as_any().get_type(), (slf.get().name(),))
    }
}
