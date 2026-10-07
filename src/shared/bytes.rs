//! Save a quantized tensor as bytes, and load it back.

use core::cmp::Ordering;
use core::mem::size_of;

use crate::error::{Error, Result, check_bits, check_block, malformed};
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
    /// [`from_bytes`](Self::from_bytes) reads exactly one tensor, so to keep
    /// several in one file, write each one's length before it.
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
            bytes.extend_from_slice(block_bits);
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
    /// [`Error::ScaleMismatch`] if the tensor was saved with another scale
    /// type, [`Error::NewerFormat`] if a newer version of quantize saved it in
    /// a format this one can't read, [`Error::Malformed`] if the bytes end
    /// early, go on past the tensor, or don't hold one, and any error from
    /// [`validate`](Self::validate).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader { bytes };
        if reader.take(MAGIC.len())? != MAGIC {
            return Err(malformed("the bytes don't start with QNTZ"));
        }
        match reader.byte()? {
            VERSION => {}
            version if version > VERSION => return Err(Error::NewerFormat { version }),
            _ => return Err(malformed("unsupported format version")),
        }
        let kind = reader.byte()?;
        let name_len = usize::from(reader.byte()?);
        let saved_name = reader.take(name_len)?;
        if saved_name != S::NAME.as_bytes() {
            return Err(Error::ScaleMismatch {
                saved: String::from_utf8_lossy(saved_name).into_owned(),
                expected: S::NAME,
            });
        }
        let code_bits = u32::from(reader.byte()?);
        let block = reader.size()?;
        let len = reader.size()?;
        let columns = Some(reader.size()?).filter(|&columns| columns > 0);

        check_block(block)?;
        let blocks = len.div_ceil(block);
        // Every kind loads as a flat vector first. The shape goes on last,
        // through `set_shape`, since it finds where each adaptive row starts
        // by reading widths that must be checked first.
        let mut quantized = match kind {
            SYMMETRIC => {
                let scales = reader.scales(blocks)?;
                let codes = reader.codes(code_bits, len)?;
                Self::Symmetric {
                    scales,
                    codes,
                    block,
                    len,
                    columns: None,
                }
            }
            ASYMMETRIC => {
                let scales = reader.scales(blocks)?;
                let zero_points = reader.scales(blocks)?;
                let codes = reader.codes(code_bits, len)?;
                Self::Asymmetric {
                    scales,
                    zero_points,
                    codes,
                    block,
                    len,
                    columns: None,
                }
            }
            ADAPTIVE => {
                let block_bits = reader.take(blocks)?.to_vec();
                let scales = reader.scales(blocks)?;
                let zero_points = reader.scales(blocks)?;
                let codes = reader.rest();
                Self::Adaptive {
                    scales,
                    zero_points,
                    codes,
                    block_bits,
                    block,
                    len,
                    columns: None,
                    row_starts: Vec::new(),
                }
            }
            _ => return Err(malformed("unknown kind")),
        };

        // The header says how many bytes the codes fill, so any other count
        // means the bytes were cut off, or something follows the tensor.
        match quantized.codes().len().cmp(&quantized.code_bytes()?) {
            Ordering::Less => return Err(malformed("the bytes end early")),
            Ordering::Greater => return Err(malformed("extra bytes follow the tensor")),
            Ordering::Equal => {}
        }
        quantized.validate()?;
        if let Some(columns) = columns {
            if !len.is_multiple_of(columns) {
                return Err(Error::ShapeMismatch { len, columns });
            }
            quantized.set_shape(len / columns, columns)?;
        }
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

    /// Everything not yet read, as `len` codes of `bits` each. A width outside
    /// `2..=16` is an error here, since [`Packed::from_raw`] panics on it.
    fn codes(self, bits: u32, len: usize) -> Result<Packed> {
        check_bits(bits)?;
        Ok(Packed::from_raw(self.rest(), bits, len))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{adaptive, asymmetric, symmetric};
    use half::{bf16, f16};

    fn values() -> Vec<f32> {
        (0..80).map(|i| (i as f32 * 0.37).sin()).collect()
    }

    fn assert_round_trips<S: Scale + PartialEq + core::fmt::Debug>() {
        let values = values();
        let mut five_by_sixteen = symmetric::quantize_with(&values, 5, 16).unwrap();
        five_by_sixteen.set_shape(5, 16).unwrap();
        let mut two_by_forty = asymmetric::quantize_with(&values, 8, 32).unwrap();
        two_by_forty.set_shape(2, 40).unwrap();
        // Its second row starts partway through a block.
        let mut adaptive_two_by_forty = adaptive::quantize_with(&values, 32, 0.01).unwrap();
        adaptive_two_by_forty.set_shape(2, 40).unwrap();
        let tensors: [Quantized<S>; 6] = [
            symmetric::quantize_with(&values, 4, 32).unwrap(),
            five_by_sixteen,
            two_by_forty,
            adaptive::quantize_with(&values, 32, 0.01).unwrap(),
            adaptive_two_by_forty,
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
    fn nbytes_is_the_saved_size_without_the_header() {
        // With f32 scales the header fills 35 bytes, as the table in
        // `to_bytes` adds up.
        let values = values();
        let tensors: [Quantized<f32>; 3] = [
            symmetric::quantize_with(&values, 5, 16).unwrap(),
            asymmetric::quantize_with(&values, 4, 32).unwrap(),
            adaptive::quantize_with(&values, 32, 0.01).unwrap(),
        ];
        for quantized in tensors {
            assert_eq!(quantized.to_bytes().len(), 35 + quantized.nbytes());
        }
    }

    #[test]
    fn every_truncation_says_the_bytes_end_early() {
        let quantized = asymmetric::quantize_with::<f16>(&values(), 4, 32).unwrap();
        let bytes = quantized.to_bytes();
        for end in 0..bytes.len() {
            assert_eq!(
                Quantized::<f16>::from_bytes(&bytes[..end]),
                Err(Error::Malformed {
                    reason: "the bytes end early"
                }),
                "{end} bytes"
            );
        }
    }

    #[test]
    fn bytes_after_the_tensor_are_an_error() {
        let quantized = adaptive::quantize_with::<f32>(&values(), 32, 0.01).unwrap();
        for extra in [vec![0], quantized.to_bytes()] {
            let mut bytes = quantized.to_bytes();
            bytes.extend(extra);
            assert_eq!(
                Quantized::<f32>::from_bytes(&bytes),
                Err(Error::Malformed {
                    reason: "extra bytes follow the tensor"
                })
            );
        }
    }

    #[test]
    fn loading_with_another_scale_type_names_both() {
        let quantized = symmetric::quantize_with::<f16>(&values(), 8, 32).unwrap();
        let bytes = quantized.to_bytes();
        assert_eq!(
            Quantized::<bf16>::from_bytes(&bytes),
            Err(Error::ScaleMismatch {
                saved: "f16".to_string(),
                expected: "bf16"
            })
        );
        assert_eq!(
            Quantized::<f32>::from_bytes(&bytes),
            Err(Error::ScaleMismatch {
                saved: "f16".to_string(),
                expected: "f32"
            })
        );
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
        assert_eq!(
            load_with(version, 2),
            Err(Error::NewerFormat { version: 2 })
        );
        assert!(matches!(
            load_with(version, 0),
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
