use std::sync::{Arc, Mutex, PoisonError};

use half::{bf16, f16};
use pyo3::prelude::*;
use quantize::{Quantized, Scale, Scheme, learned};

use super::parts::Parts;
use crate::error::from_quantize;
use crate::scale::PyScale;

/// The tensor, with one of three scale types, behind an `Arc` so that a
/// clone shares its codes and scales instead of copying them. `refine` and
/// `alternate` change it through `Arc::make_mut`, which copies a shared
/// tensor first, so a `copy()`, or a `dot`, `matmul`, or `dequantize` still
/// running, keeps the values it had.
#[derive(Clone, PartialEq)]
pub(crate) enum QuantizedInner {
    F32(Arc<Quantized<f32>>),
    F16(Arc<Quantized<f16>>),
    Bf16(Arc<Quantized<bf16>>),
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
    let mut quantized = scheme.quantize(values)?;
    if let [rows, columns] = *shape {
        quantized.set_shape(rows, columns)?;
    }
    Ok(quantized)
}

impl QuantizedInner {
    fn from_scheme(
        scheme: Scheme,
        values: &[f32],
        shape: &[usize],
        scale: PyScale,
    ) -> PyResult<Self> {
        match scale {
            PyScale::F32 => quantize_shaped(scheme, values, shape)
                .map(Arc::new)
                .map(Self::F32),
            PyScale::F16 => quantize_shaped(scheme, values, shape)
                .map(Arc::new)
                .map(Self::F16),
            PyScale::Bf16 => quantize_shaped(scheme, values, shape)
                .map(Arc::new)
                .map(Self::Bf16),
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
        with_inner!(self, |quantized| {
            learned::refine(Arc::make_mut(quantized), values)
        })
    }

    pub(crate) fn alternate(&mut self, values: &[f32]) -> quantize::Result<bool> {
        with_inner!(self, |quantized| {
            learned::alternate(Arc::make_mut(quantized), values)
        })
    }

    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        with_inner!(self, |quantized| quantized.to_bytes())
    }

    /// Load bytes that [`Quantized::to_bytes`] saved, with the scale type
    /// their header names.
    pub(crate) fn from_bytes(bytes: &[u8]) -> quantize::Result<Self> {
        match saved_scale(bytes) {
            PyScale::F32 => Quantized::from_bytes(bytes).map(Arc::new).map(Self::F32),
            PyScale::F16 => Quantized::from_bytes(bytes).map(Arc::new).map(Self::F16),
            PyScale::Bf16 => Quantized::from_bytes(bytes).map(Arc::new).map(Self::Bf16),
        }
    }

    pub(crate) fn from_parts(parts: Parts, scale: PyScale) -> PyResult<Self> {
        match scale {
            PyScale::F32 => parts.into_quantized().map(Arc::new).map(Self::F32),
            PyScale::F16 => parts.into_quantized().map(Arc::new).map(Self::F16),
            PyScale::Bf16 => parts.into_quantized().map(Arc::new).map(Self::Bf16),
        }
    }
}

/// Quantized values: small integer codes, with one scale for each block of
/// `block` values, and one zero-point too for asymmetric and adaptive ones.
/// `len(q)` is the number of values, `rows * columns` for a matrix, as in
/// Rust, and `shape` gives the rows and columns. Value `i` is in block
/// `i // block`, counting a matrix row after row, and decodes as
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
///
/// `Quantized(data)` loads a tensor that `to_bytes` saved, like `from_bytes`.
/// Pickles load through it, so `torch.load` accepts them once
/// `torch.serialization.add_safe_globals([Quantized])` allows the class.
#[pyclass(name = "Quantized", module = "quantize", eq, frozen)]
pub struct PyQuantized {
    // `frozen` drops PyO3's borrow flag, which makes a refit and another
    // thread's read fail when they overlap without the GIL. Threads share the
    // tensor through this lock instead, held only to clone the `Arc` or to put
    // a refit in its place, never while computing or calling Python. Each
    // store replaces the tensor in one step, so even a lock that a panic
    // poisoned holds a whole tensor.
    inner: Mutex<QuantizedInner>,
}

impl PyQuantized {
    pub fn from_scheme(
        scheme: Scheme,
        values: &[f32],
        shape: &[usize],
        scale: PyScale,
    ) -> PyResult<Self> {
        QuantizedInner::from_scheme(scheme, values, shape, scale).map(Self::from)
    }

    pub fn len(&self) -> usize {
        self.snapshot().len()
    }

    /// The tensor as it is now, sharing its codes and scales. A refit stored
    /// later puts a new tensor in its place, so this one keeps its values.
    pub(crate) fn snapshot(&self) -> QuantizedInner {
        self.inner
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Put `inner`, such as a refit of a snapshot, in place of the tensor.
    pub(crate) fn store(&self, inner: QuantizedInner) {
        *self.inner.lock().unwrap_or_else(PoisonError::into_inner) = inner;
    }
}

impl From<QuantizedInner> for PyQuantized {
    fn from(inner: QuantizedInner) -> Self {
        Self {
            inner: Mutex::new(inner),
        }
    }
}

impl PartialEq for PyQuantized {
    fn eq(&self, other: &Self) -> bool {
        self.snapshot() == other.snapshot()
    }
}
