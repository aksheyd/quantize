//! Save a quantized tensor as bytes, and load it back.

use core::mem::size_of;

use crate::error::{check_block, malformed, Result};
use crate::packed::Packed;
use crate::scale::Scale;
use crate::tensor::Quantized;

/// The first bytes of every saved tensor.
const MAGIC: &[u8; 4] = b"QNTZ";
/// Changes whenever the layout does.
const VERSION: u8 = 1;

const SYMMETRIC: u8 = 0;
const ASYMMETRIC: u8 = 1;
const ADAPTIVE: u8 = 2;

impl<S: Scale> Quantized<S> {
    /// Save the tensor as bytes that [`from_bytes`](Self::from_bytes) reads
    /// back. Sizes are little-endian `u64`s:
    ///
    /// | bytes | field |
    /// | --- | --- |
    /// | 4 | `QNTZ` |
    /// | 1 | format version: 1 |
    /// | 1 | kind: 0 symmetric, 1 asymmetric, 2 adaptive |
    /// | 1 | `n`, the length of the scale type's [`NAME`](Scale::NAME) |
    /// | `n` | the name: `f32`, `f16`, or `bf16` |
    /// | 1 | code bits, 2 to 16, or 0 for adaptive |
    /// | 8 | `block` |
    /// | 8 | `len` |
    /// | 8 | `columns`, or 0 for a flat vector |
    /// | 1 per block | adaptive only: the block's bit width |
    /// | 4 or 2 per block | scales, as little-endian `f32`, `f16`, or `bf16` |
    /// | 4 or 2 per block | asymmetric and adaptive only: zero-points |
    /// | the rest | the packed [`codes`](Self::codes) |
    ///
    /// ```
    /// use quantize::{quantize, Quantized};
    ///
    /// let q = quantize::<f32, 8, 32>(&[0.42, -0.10, 0.70, -0.50]).unwrap();
    /// let bytes = q.to_bytes();
    /// assert_eq!(Quantized::<f32>::from_bytes(&bytes).unwrap(), q);
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        let (kind, code_bits) = match self {
            Self::Symmetric { codes, .. } => (SYMMETRIC, codes.bits()),
            Self::Asymmetric { codes, .. } => (ASYMMETRIC, codes.bits()),
            Self::Adaptive { .. } => (ADAPTIVE, 0),
        };
        let columns = self.shape().map_or(0, |(_, columns)| columns);

        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.push(VERSION);
        bytes.push(kind);
        bytes.push(S::NAME.len() as u8);
        bytes.extend_from_slice(S::NAME.as_bytes());
        bytes.push(code_bits as u8);
        for size in [self.block(), self.len(), columns] {
            bytes.extend_from_slice(&(size as u64).to_le_bytes());
        }
        if let Some(block_bits) = self.block_bits() {
            bytes.extend(block_bits.iter().map(|&bits| bits as u8));
        }
        for &value in self.scales().iter().chain(self.zero_points()) {
            value.write_le_bytes(&mut bytes);
        }
        bytes.extend_from_slice(self.codes());
        bytes
    }

    /// Load a tensor that [`to_bytes`](Self::to_bytes) saved with the same
    /// scale type `S`. The result passes [`validate`](Self::validate), so
    /// truncated or corrupted bytes are an error, never a tensor that decodes
    /// out of bounds.
    ///
    /// # Errors
    ///
    /// [`Error::Malformed`](crate::Error::Malformed) if the bytes don't hold
    /// a tensor saved with scale type `S`, or any error from
    /// [`validate`](Self::validate).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader { bytes };
        if reader.take(MAGIC.len())? != MAGIC {
            return Err(malformed("the bytes don't start with QNTZ"));
        }
        if reader.byte()? != VERSION {
            return Err(malformed("unsupported format version"));
        }
        let kind = reader.byte()?;
        let name_len = usize::from(reader.byte()?);
        if reader.take(name_len)? != S::NAME.as_bytes() {
            return Err(malformed("the tensor was saved with another scale type"));
        }
        let code_bits = u32::from(reader.byte()?);
        let block = reader.size()?;
        let len = reader.size()?;
        let columns = Some(reader.size()?).filter(|&columns| columns > 0);

        check_block(block)?;
        let blocks = len.div_ceil(block);
        let quantized = match kind {
            SYMMETRIC => {
                let scales = reader.scales(blocks)?;
                let codes = Packed::from_raw(reader.rest(), code_bits, len);
                Self::Symmetric {
                    scales,
                    codes,
                    block,
                    len,
                    columns,
                }
            }
            ASYMMETRIC => {
                let scales = reader.scales(blocks)?;
                let zero_points = reader.scales(blocks)?;
                let codes = Packed::from_raw(reader.rest(), code_bits, len);
                Self::Asymmetric {
                    scales,
                    zero_points,
                    codes,
                    block,
                    len,
                    columns,
                }
            }
            ADAPTIVE => {
                let bits = reader
                    .take(blocks)?
                    .iter()
                    .map(|&bits| bits.into())
                    .collect();
                let scales = reader.scales(blocks)?;
                let zero_points = reader.scales(blocks)?;
                let bytes = reader.rest();
                Self::Adaptive {
                    scales,
                    zero_points,
                    bytes,
                    bits,
                    block,
                    len,
                    columns,
                }
            }
            _ => return Err(malformed("unknown kind")),
        };
        quantized.validate()?;
        Ok(quantized)
    }
}

/// Reads the fields that [`Quantized::to_bytes`] wrote, in order.
struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let (taken, rest) = self
            .bytes
            .split_at_checked(count)
            .ok_or(malformed("the bytes end early"))?;
        self.bytes = rest;
        Ok(taken)
    }

    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    /// A little-endian `u64` that must fit in a `usize`.
    fn size(&mut self) -> Result<usize> {
        let mut little_endian = [0; 8];
        little_endian.copy_from_slice(self.take(8)?);
        usize::try_from(u64::from_le_bytes(little_endian))
            .map_err(|_| malformed("a size is too large for this platform"))
    }

    fn scales<S: Scale>(&mut self, count: usize) -> Result<Vec<S>> {
        let width = size_of::<S>();
        let total = count
            .checked_mul(width)
            .ok_or(malformed("the bytes end early"))?;
        let bytes = self.take(total)?;
        Ok(bytes.chunks_exact(width).map(S::read_le_bytes).collect())
    }

    /// Everything not yet read.
    fn rest(self) -> Vec<u8> {
        self.bytes.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use crate::{adaptive, asymmetric, symmetric};
    use half::{bf16, f16};

    fn values() -> Vec<f32> {
        (0..80).map(|i| (i as f32 * 0.37).sin()).collect()
    }

    fn assert_round_trips<S: Scale + PartialEq + core::fmt::Debug>() {
        let values = values();
        let tensors: [Quantized<S>; 5] = [
            symmetric::quantize_with(&values, 4, 32).unwrap(),
            symmetric::quantize_with(&values, 5, 16)
                .and_then(|quantized| quantized.into_matrix(5, 16))
                .unwrap(),
            asymmetric::quantize_with(&values, 8, 32)
                .and_then(|quantized| quantized.into_matrix(2, 40))
                .unwrap(),
            adaptive::quantize_with(&values, 32, 0.01).unwrap(),
            symmetric::quantize_with(&[], 8, 32).unwrap(),
        ];
        for quantized in tensors {
            let loaded = Quantized::<S>::from_bytes(&quantized.to_bytes());
            assert_eq!(loaded, Ok(quantized));
        }
    }

    #[test]
    fn every_kind_and_scale_type_round_trips() {
        assert_round_trips::<f32>();
        assert_round_trips::<f16>();
        assert_round_trips::<bf16>();
    }

    #[test]
    fn every_truncation_is_an_error() {
        let quantized = asymmetric::quantize_with::<f16>(&values(), 4, 32).unwrap();
        let bytes = quantized.to_bytes();
        for end in 0..bytes.len() {
            let loaded = Quantized::<f16>::from_bytes(&bytes[..end]);
            assert!(loaded.is_err(), "{end} bytes");
        }
    }

    #[test]
    fn an_extra_byte_is_an_error() {
        let quantized = adaptive::quantize_with::<f32>(&values(), 32, 0.01).unwrap();
        let mut bytes = quantized.to_bytes();
        bytes.push(0);
        let loaded = Quantized::<f32>::from_bytes(&bytes);
        assert!(matches!(loaded, Err(Error::Malformed { .. })));
    }

    #[test]
    fn loading_with_another_scale_type_is_an_error() {
        let quantized = symmetric::quantize_with::<f16>(&values(), 8, 32).unwrap();
        let bytes = quantized.to_bytes();
        let as_bf16 = Quantized::<bf16>::from_bytes(&bytes);
        let as_f32 = Quantized::<f32>::from_bytes(&bytes);
        assert!(matches!(as_bf16, Err(Error::Malformed { .. })));
        assert!(matches!(as_f32, Err(Error::Malformed { .. })));
    }

    #[test]
    fn header_fields_are_checked() {
        let quantized = symmetric::quantize_with::<f32>(&values(), 8, 32).unwrap();
        let bytes = quantized.to_bytes();
        // Offsets follow the table in `to_bytes`; the name "f32" takes 3 bytes.
        let (version, kind, code_bits, block, columns) = (4, 5, 10, 11, 27);
        let load_with = |offset: usize, value: u8| {
            let mut changed = bytes.clone();
            changed[offset] = value;
            Quantized::<f32>::from_bytes(&changed)
        };
        assert!(matches!(load_with(0, b'X'), Err(Error::Malformed { .. })));
        assert!(matches!(
            load_with(version, 2),
            Err(Error::Malformed { .. })
        ));
        assert!(matches!(load_with(kind, 3), Err(Error::Malformed { .. })));
        assert_eq!(load_with(code_bits, 1), Err(Error::InvalidBits { bits: 1 }));
        assert_eq!(load_with(block, 0), Err(Error::InvalidBlock { block: 0 }));
        assert_eq!(
            load_with(columns, 7),
            Err(Error::ShapeMismatch {
                len: 80,
                columns: 7
            })
        );
    }

    #[test]
    fn validate_rejects_codes_too_short_for_len() {
        let short = Quantized::<f32>::Symmetric {
            scales: vec![1.0; 2],
            codes: Packed::from_raw(vec![0; 16], 4, 64),
            block: 32,
            len: 64,
            columns: None,
        };
        assert!(matches!(short.validate(), Err(Error::Malformed { .. })));
    }
}
