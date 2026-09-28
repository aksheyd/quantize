use half::{bf16, f16};
use pyo3::prelude::*;
use quantize::{learned, Quantized, Scale, Scheme};

use super::parts::Parts;
use crate::error::from_quantize;
use crate::scale::PyScale;

#[derive(Clone, PartialEq)]
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

/// The scale type that [`Quantized::to_bytes`] saved `bytes` with. `QNTZ`,
/// the format version, and the kind fill the first 6 bytes, then one byte
/// gives the length of the scale type's [`Scale::NAME`], which follows it.
/// Bytes that name no scale type read as f32, so that
/// [`Quantized::from_bytes`] says what's wrong with them.
fn saved_scale(bytes: &[u8]) -> PyScale {
    let name = bytes
        .get(6)
        .and_then(|&length| bytes.get(7..7 + usize::from(length)));
    if name == Some(f16::NAME.as_bytes()) {
        PyScale::F16
    } else if name == Some(bf16::NAME.as_bytes()) {
        PyScale::Bf16
    } else {
        PyScale::F32
    }
}

/// Quantize `values`, then record `shape` if it is a matrix's.
fn quantize_shaped<S: Scale>(
    scheme: Scheme,
    values: &[f32],
    shape: &[usize],
) -> quantize::Result<Quantized<S>> {
    let quantized = scheme.quantize(values)?;
    match *shape {
        [rows, columns] => quantized.into_matrix(rows, columns),
        _ => Ok(quantized),
    }
}

impl QuantizedInner {
    fn from_scheme(
        scheme: Scheme,
        values: &[f32],
        shape: &[usize],
        scale: PyScale,
    ) -> PyResult<Self> {
        match scale {
            PyScale::F32 => quantize_shaped(scheme, values, shape).map(Self::F32),
            PyScale::F16 => quantize_shaped(scheme, values, shape).map(Self::F16),
            PyScale::Bf16 => quantize_shaped(scheme, values, shape).map(Self::Bf16),
        }
        .map_err(from_quantize)
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

    /// The NumPy shape: `[rows, columns]` for a matrix, or `[len]`.
    pub(crate) fn shape(&self) -> Vec<usize> {
        match with_inner!(self, |quantized| quantized.shape()) {
            Some((rows, columns)) => vec![rows, columns],
            None => vec![self.len()],
        }
    }

    pub(crate) fn refine(&mut self, values: &[f32]) -> quantize::Result<()> {
        with_inner!(self, |quantized| learned::refine(quantized, values))
    }

    pub(crate) fn alternate(&mut self, values: &[f32]) -> quantize::Result<()> {
        with_inner!(self, |quantized| learned::alternate(quantized, values))
    }

    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        with_inner!(self, |quantized| quantized.to_bytes())
    }

    /// Load bytes that [`Quantized::to_bytes`] saved, with the scale type
    /// their header names.
    pub(crate) fn from_bytes(bytes: &[u8]) -> quantize::Result<Self> {
        match saved_scale(bytes) {
            PyScale::F32 => Quantized::from_bytes(bytes).map(Self::F32),
            PyScale::F16 => Quantized::from_bytes(bytes).map(Self::F16),
            PyScale::Bf16 => Quantized::from_bytes(bytes).map(Self::Bf16),
        }
    }

    pub(crate) fn from_parts(parts: Parts, scale: PyScale) -> PyResult<Self> {
        match scale {
            PyScale::F32 => parts.into_quantized().map(Self::F32),
            PyScale::F16 => parts.into_quantized().map(Self::F16),
            PyScale::Bf16 => parts.into_quantized().map(Self::Bf16),
        }
    }
}

/// Quantized values: small integer codes, with one scale for each block of
/// `block` values, and one zero-point too for asymmetric and adaptive ones.
/// Value `i` is in block `i // block`, counting a matrix row after row, and
/// decodes as
///
///     code * scale                   (symmetric)
///     (code - zero_point) * scale    (asymmetric and adaptive)
///
/// Codes are signed and `bits` wide: -8 to 7 at 4 bits. `codes` packs them
/// low bits first, so at 4 bits the first code of each byte is its low
/// nibble, and `unpacked_codes` gives one per value. An adaptive tensor packs
/// each block at its own width from `block_bits`, starting on a new byte.
///
/// Scales can be negative: a symmetric block puts its value farthest from
/// zero on the most negative code, even when that value is positive.
/// Zero-points are rarely whole numbers.
#[pyclass(name = "Quantized", module = "quantize", eq)]
#[derive(PartialEq)]
pub struct PyQuantized {
    pub(crate) inner: QuantizedInner,
}

impl PyQuantized {
    pub fn from_scheme(
        scheme: Scheme,
        values: &[f32],
        shape: &[usize],
        scale: PyScale,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: QuantizedInner::from_scheme(scheme, values, shape, scale)?,
        })
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn refine(&mut self, values: &[f32]) -> quantize::Result<()> {
        self.inner.refine(values)
    }

    pub fn alternate(&mut self, values: &[f32]) -> quantize::Result<()> {
        self.inner.alternate(values)
    }

    pub(crate) fn dequantize_into(&self, out: &mut [f32]) -> quantize::Result<()> {
        with_inner!(&self.inner, |quantized| quantized.dequantize_into(out))
    }
}
