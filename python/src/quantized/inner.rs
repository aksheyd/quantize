use half::{bf16, f16};
use pyo3::prelude::*;
use quantize::{learned, Quantized, Scheme};

use crate::error::from_quantize;
use crate::scale::PyScale;

#[derive(Clone)]
pub(crate) enum QuantizedInner {
    F32(Quantized<f32>),
    F16(Quantized<f16>),
    Bf16(Quantized<bf16>),
}

macro_rules! with_inner {
    ($inner:expr, |$quantized:ident| $body:expr) => {
        match $inner {
            $crate::quantized::inner::QuantizedInner::F32($quantized) => $body,
            $crate::quantized::inner::QuantizedInner::F16($quantized) => $body,
            $crate::quantized::inner::QuantizedInner::Bf16($quantized) => $body,
        }
    };
}

pub(crate) use with_inner;

impl QuantizedInner {
    fn from_scheme(scheme: Scheme, values: &[f32], scale: PyScale) -> PyResult<Self> {
        match scale {
            PyScale::F32 => scheme
                .quantize(values)
                .map(Self::F32)
                .map_err(from_quantize),
            PyScale::F16 => scheme
                .quantize(values)
                .map(Self::F16)
                .map_err(from_quantize),
            PyScale::Bf16 => scheme
                .quantize(values)
                .map(Self::Bf16)
                .map_err(from_quantize),
        }
    }

    pub(crate) fn scale(&self) -> PyScale {
        match self {
            Self::F32(_) => PyScale::F32,
            Self::F16(_) => PyScale::F16,
            Self::Bf16(_) => PyScale::Bf16,
        }
    }

    pub(crate) fn len(&self) -> usize {
        with_inner!(self, |quantized| quantized.len())
    }

    pub(crate) fn refine(&mut self, values: &[f32]) {
        with_inner!(self, |quantized| learned::refine(quantized, values));
    }
}

#[pyclass(name = "Quantized", module = "quantize")]
pub struct PyQuantized {
    pub(crate) inner: QuantizedInner,
}

impl PyQuantized {
    pub fn from_scheme(scheme: Scheme, values: &[f32], scale: PyScale) -> PyResult<Self> {
        Ok(Self {
            inner: QuantizedInner::from_scheme(scheme, values, scale)?,
        })
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn refine(&mut self, values: &[f32]) {
        self.inner.refine(values);
    }

    pub(crate) fn dequantize_into(&self, out: &mut [f32]) -> quantize::Result<()> {
        with_inner!(&self.inner, |quantized| quantized.dequantize_into(out))
    }
}
