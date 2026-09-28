//! Rebuild a tensor from the values its getters return.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use quantize::{Error, Packed, Quantized, Scale};

use crate::error::from_quantize;

const SHAPE_DIMENSIONS: &str = "shape must be (len,) or (rows, columns)";
const SHAPE_TOO_LARGE: &str = "the shape holds too many values";
const KIND_PARTS: &str = "kind must be 'symmetric' or 'asymmetric', with bits, or 'adaptive', with block_bits instead; symmetric tensors take no zero_points";

/// The values that `Quantized`'s getters of the same names return, with the
/// arrays read into vectors.
pub(crate) struct Parts {
    pub(crate) kind: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) block: usize,
    pub(crate) codes: Vec<u8>,
    pub(crate) scales: Vec<f32>,
    pub(crate) zero_points: Vec<f32>,
    pub(crate) bits: Option<u32>,
    pub(crate) block_bits: Option<Vec<u32>>,
}

impl Parts {
    /// Build the variant that `kind` names, then [`Quantized::validate`] it,
    /// so parts that don't fit together are an error instead of a tensor
    /// that decodes out of bounds.
    pub(crate) fn into_quantized<S: Scale>(self) -> PyResult<Quantized<S>> {
        let (len, columns) = match self.shape[..] {
            [len] => (len, None),
            [rows, columns] => match rows.checked_mul(columns) {
                Some(len) => (len, Some(columns)),
                None => return Err(PyValueError::new_err(SHAPE_TOO_LARGE)),
            },
            _ => return Err(PyValueError::new_err(SHAPE_DIMENSIONS)),
        };
        let block = self.block;
        let scales: Vec<S> = self.scales.into_iter().map(S::from_f32).collect();
        let zero_points: Vec<S> = self.zero_points.into_iter().map(S::from_f32).collect();

        let quantized = match (self.kind.as_str(), self.bits, self.block_bits) {
            ("symmetric", Some(bits), None) if zero_points.is_empty() => Quantized::Symmetric {
                scales,
                codes: packed_codes(self.codes, bits, len)?,
                block,
                len,
                columns,
            },
            ("asymmetric", Some(bits), None) => Quantized::Asymmetric {
                scales,
                zero_points,
                codes: packed_codes(self.codes, bits, len)?,
                block,
                len,
                columns,
            },
            ("adaptive", None, Some(block_bits)) => Quantized::Adaptive {
                scales,
                zero_points,
                codes: self.codes,
                block_bits: block_widths(block_bits)?,
                block,
                len,
                columns,
            },
            _ => return Err(PyValueError::new_err(KIND_PARTS)),
        };
        quantized.validate().map_err(from_quantize)?;
        Ok(quantized)
    }
}

/// Each block's width in a byte. A width too large for one is far outside 2
/// to 16, so it raises `InvalidBitsError`, as 17 does.
fn block_widths(block_bits: Vec<u32>) -> PyResult<Vec<u8>> {
    block_bits
        .into_iter()
        .map(|bits| u8::try_from(bits).map_err(|_| from_quantize(Error::InvalidBits { bits })))
        .collect()
}

/// The codes, packed at `bits` each. A width outside 2 to 16 raises
/// `InvalidBitsError`, since `Packed::from_raw` panics on it.
fn packed_codes(codes: Vec<u8>, bits: u32, len: usize) -> PyResult<Packed> {
    if !(2..=16).contains(&bits) {
        return Err(from_quantize(Error::InvalidBits { bits }));
    }
    Ok(Packed::from_raw(codes, bits, len))
}
