//! One enum, one variant per scheme.

use crate::decode::{decode_row, dequant_adaptive, dequant_asym, dequant_sym, dot_of, matmul_into};
use crate::error::{Error, Result, check_bits, check_block, check_len, malformed};
use crate::packed::Packed;
use crate::scale::Scale;

/// Packed codes and the scheme that produced them.
///
/// Value `i` is in block `i / block`, counting a matrix row after row, and
/// decodes from its code and that block's scale: as `code * scale` in a
/// symmetric block, and as `(code - zero_point) * scale` in an asymmetric or
/// adaptive one. Scales can be negative: a symmetric block puts its value
/// farthest from zero on the most negative code, even when that value is
/// positive, as [`symmetric_scale`](crate::params::symmetric_scale) explains.
///
/// The `len` values are a flat vector until
/// [`set_shape`](Self::set_shape) records `columns`, the length of each
/// row of a row-major matrix.
#[derive(Clone, PartialEq)]
pub enum Quantized<S: Scale> {
    /// One scale per block.
    Symmetric {
        scales: Vec<S>,
        codes: Packed,
        block: usize,
        len: usize,
        columns: Option<usize>,
    },
    /// Scale and zero-point per block.
    Asymmetric {
        scales: Vec<S>,
        zero_points: Vec<S>,
        codes: Packed,
        block: usize,
        len: usize,
        columns: Option<usize>,
    },
    /// Per-block bit width; codes packed at that width.
    Adaptive {
        scales: Vec<S>,
        zero_points: Vec<S>,
        bytes: Vec<u8>,
        bits: Vec<u8>,
        block: usize,
        len: usize,
        columns: Option<usize>,
    },
}

impl<S: Scale> Quantized<S> {
    pub fn len(&self) -> usize {
        match self {
            Self::Symmetric { len, .. }
            | Self::Asymmetric { len, .. }
            | Self::Adaptive { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `(rows, columns)` once [`set_shape`](Self::set_shape) has recorded a
    /// shape, or `None` for a flat vector.
    pub fn shape(&self) -> Option<(usize, usize)> {
        let columns = match self {
            Self::Symmetric { columns, .. }
            | Self::Asymmetric { columns, .. }
            | Self::Adaptive { columns, .. } => (*columns)?,
        };
        Some((self.len().checked_div(columns)?, columns))
    }

    /// Read the values as a row-major `rows × columns` matrix: the first
    /// `columns` values are row 0, the next `columns` are row 1, and so on.
    /// [`matmul`](Self::matmul) multiplies by that matrix. For a linear
    /// layer's weights, `rows` is the number of outputs and `columns` the
    /// number of inputs, as in PyTorch's `Linear.weight`.
    ///
    /// # Errors
    ///
    /// [`Error::ShapeMismatch`] if `columns` is 0, and
    /// [`Error::MatrixMismatch`] if `rows × columns` isn't
    /// [`len`](Self::len). Either way the tensor is left as it was.
    pub fn set_shape(&mut self, rows: usize, columns: usize) -> Result<()> {
        let len = self.len();
        if columns == 0 {
            return Err(Error::ShapeMismatch { len, columns });
        }
        if rows.checked_mul(columns) != Some(len) {
            return Err(Error::MatrixMismatch { rows, columns, len });
        }
        match self {
            Self::Symmetric {
                columns: recorded, ..
            }
            | Self::Asymmetric {
                columns: recorded, ..
            }
            | Self::Adaptive {
                columns: recorded, ..
            } => *recorded = Some(columns),
        }
        Ok(())
    }

    pub fn block(&self) -> usize {
        match self {
            Self::Symmetric { block, .. }
            | Self::Asymmetric { block, .. }
            | Self::Adaptive { block, .. } => *block,
        }
    }

    pub fn scales(&self) -> &[S] {
        match self {
            Self::Symmetric { scales, .. }
            | Self::Asymmetric { scales, .. }
            | Self::Adaptive { scales, .. } => scales,
        }
    }

    pub fn zero_points(&self) -> &[S] {
        match self {
            Self::Symmetric { .. } => &[],
            Self::Asymmetric { zero_points, .. } | Self::Adaptive { zero_points, .. } => {
                zero_points
            }
        }
    }

    pub fn codes(&self) -> &[u8] {
        match self {
            Self::Symmetric { codes, .. } | Self::Asymmetric { codes, .. } => codes.as_bytes(),
            Self::Adaptive { bytes, .. } => bytes,
        }
    }

    pub fn block_bits(&self) -> Option<&[u8]> {
        match self {
            Self::Adaptive { bits, .. } => Some(bits),
            _ => None,
        }
    }

    pub fn nbytes(&self) -> usize {
        let extra = match self {
            Self::Adaptive { bits, .. } => core::mem::size_of_val(bits.as_slice()),
            _ => 0,
        };
        self.codes().len()
            + core::mem::size_of_val(self.scales())
            + core::mem::size_of_val(self.zero_points())
            + extra
    }

    pub fn bits_per_element(&self) -> f32 {
        if self.is_empty() {
            0.0
        } else {
            self.nbytes() as f32 * 8.0 / self.len() as f32
        }
    }

    /// Check that the buffers are exactly as long as `block`, `len`, the bit
    /// widths, and the shape say, since decoding relies on it.
    /// [`from_bytes`](Self::from_bytes) runs this; so should code that builds
    /// a variant by hand.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidBlock`], [`Error::InvalidBits`], or
    /// [`Error::ShapeMismatch`] for a field out of range, and
    /// [`Error::Malformed`] for a buffer of the wrong length.
    pub fn validate(&self) -> Result<()> {
        let (len, block) = (self.len(), self.block());
        check_block(block)?;
        let columns = match self {
            Self::Symmetric { columns, .. }
            | Self::Asymmetric { columns, .. }
            | Self::Adaptive { columns, .. } => *columns,
        };
        if let Some(columns) = columns
            && (columns == 0 || !len.is_multiple_of(columns))
        {
            return Err(Error::ShapeMismatch { len, columns });
        }

        let blocks = len.div_ceil(block);
        let zero_points = match self {
            Self::Symmetric { .. } => 0,
            Self::Asymmetric { .. } | Self::Adaptive { .. } => blocks,
        };
        if self.scales().len() != blocks || self.zero_points().len() != zero_points {
            return Err(malformed(
                "every block needs one scale, and one zero-point unless symmetric",
            ));
        }

        match self {
            Self::Symmetric { codes, .. } | Self::Asymmetric { codes, .. } => {
                if codes.len() != len {
                    return Err(malformed("the packed codes must hold len values"));
                }
            }
            Self::Adaptive { bits, .. } => {
                if bits.len() != blocks {
                    return Err(malformed("every block needs one bit width"));
                }
            }
        }
        if self.codes().len() != self.code_bytes()? {
            return Err(malformed(
                "the codes must fill exactly the bytes that len and the bit widths need",
            ));
        }
        Ok(())
    }

    /// Bytes that the codes should fill: `len` codes at one bit width, or each
    /// adaptive block's codes at that block's width. An adaptive tensor must
    /// have one bit width per block.
    pub(crate) fn code_bytes(&self) -> Result<usize> {
        match self {
            Self::Symmetric { codes, .. } | Self::Asymmetric { codes, .. } => {
                packed_size(self.len(), codes.bits())
            }
            Self::Adaptive {
                bits, block, len, ..
            } => {
                let mut total = 0_usize;
                for (block_index, &bit_width) in bits.iter().enumerate() {
                    let count = (*block).min(len - block_index * block);
                    let block_bytes = packed_size(count, bit_width.into())?;
                    total = total.checked_add(block_bytes).ok_or(too_large())?;
                }
                Ok(total)
            }
        }
    }

    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = vec![0.0; self.len()];
        let _ = self.dequantize_into(&mut out);
        out
    }

    pub fn dequantize_into(&self, out: &mut [f32]) -> Result<()> {
        check_len(self.len(), out.len())?;
        if self.is_empty() {
            return Ok(());
        }
        match self {
            Self::Symmetric {
                scales,
                codes,
                block,
                ..
            } => dequant_sym(scales, codes, *block, out),
            Self::Asymmetric {
                scales,
                zero_points,
                codes,
                block,
                ..
            } => dequant_asym(scales, zero_points, codes, *block, out),
            Self::Adaptive {
                scales,
                zero_points,
                bytes,
                bits,
                block,
                len,
                ..
            } => dequant_adaptive(scales, zero_points, bytes, bits, *block, *len, out),
        }
        Ok(())
    }

    /// Decode row `row` of the matrix into `out`, without decoding the other
    /// rows. An embedding table keeps one row per token, so looking up a
    /// token decodes just its row:
    ///
    /// ```
    /// use quantize::quantize;
    ///
    /// // 3 tokens, 4 values each.
    /// let mut table = quantize::<f32, 8, 4>(&[0.1, 0.2, 0.3, 0.4,
    ///                                         0.5, 0.6, 0.7, 0.8,
    ///                                         0.9, 1.0, 1.1, 1.2]).unwrap();
    /// table.set_shape(3, 4).unwrap();
    ///
    /// let mut embedding = [0.0; 4];
    /// table.dequantize_row(1, &mut embedding).unwrap();
    /// assert_eq!(embedding, [0.5, 0.6, 0.7, 0.8]);
    /// ```
    ///
    /// # Errors
    ///
    /// [`Error::NotAMatrix`] if [`set_shape`](Self::set_shape) hasn't
    /// recorded a shape, [`Error::RowOutOfRange`] if `row` isn't below `rows`,
    /// and [`Error::LengthMismatch`] if `out` isn't `columns` long.
    pub fn dequantize_row(&self, row: usize, out: &mut [f32]) -> Result<()> {
        let Some((rows, columns)) = self.shape() else {
            return Err(Error::NotAMatrix { len: self.len() });
        };
        if row >= rows {
            return Err(Error::RowOutOfRange { row, rows });
        }
        check_len(columns, out.len())?;
        decode_row(self, row, out);
        Ok(())
    }

    pub fn dot(&self, rhs: &[f32]) -> Result<f32> {
        check_len(self.len(), rhs.len())?;
        Ok(if self.is_empty() {
            0.0
        } else {
            dot_of(self, rhs)
        })
    }

    /// Multiply a batch of input vectors by this tensor's matrix.
    ///
    /// - The tensor is the row-major `rows × columns` matrix `W` that
    ///   [`set_shape`](Self::set_shape) recorded.
    /// - `inputs` holds `batch` vectors of `columns` values each, back to back.
    /// - The result holds `batch × rows` values, row-major: value
    ///   `b * rows + r` is input `b` dotted with row `r` of `W`. That is
    ///   `inputs · Wᵀ`, which is what a linear layer computes.
    ///
    /// ```
    /// use quantize::quantize;
    ///
    /// // W has 2 rows × 3 columns.
    /// let mut w = quantize::<f32, 8, 3>(&[1.0, 0.0, 0.0,
    ///                                     0.0, 1.0, 1.0]).unwrap();
    /// w.set_shape(2, 3).unwrap();
    ///
    /// // A batch of 2 inputs, 3 values each.
    /// let inputs = [1.0, 2.0, 3.0,
    ///               4.0, 5.0, 6.0];
    ///
    /// // 2 inputs × 2 rows.
    /// let out = w.matmul(&inputs).unwrap();
    /// assert_eq!(out, [1.0, 5.0,
    ///                  4.0, 11.0]);
    /// ```
    ///
    /// Each row is decoded straight from the packed codes when `columns` is a
    /// multiple of the block size and each row fills whole bytes. Otherwise,
    /// and for adaptive tensors, every call first decodes the whole matrix.
    ///
    /// # Errors
    ///
    /// [`Error::NotAMatrix`] if the tensor has no shape,
    /// [`Error::ShapeMismatch`] if `inputs` doesn't split into whole vectors
    /// of `columns` values, and [`Error::OutputTooLarge`] if the
    /// `batch × rows` result can't be allocated.
    pub fn matmul(&self, inputs: &[f32]) -> Result<Vec<f32>> {
        let Some((rows, columns)) = self.shape() else {
            return Err(Error::NotAMatrix { len: self.len() });
        };
        if !inputs.len().is_multiple_of(columns) {
            return Err(Error::ShapeMismatch {
                len: inputs.len(),
                columns,
            });
        }
        let batch = inputs.len() / columns;

        // The result can be far larger than the tensor and the inputs
        // together. Multiply with an overflow check and reserve the memory up
        // front, so a result too big for memory is an error instead of an
        // abort.
        let too_large = Error::OutputTooLarge { batch, rows };
        let Some(output_len) = batch.checked_mul(rows) else {
            return Err(too_large);
        };
        let mut out = Vec::new();
        if out.try_reserve_exact(output_len).is_err() {
            return Err(too_large);
        }
        out.resize(output_len, 0.0);

        matmul_into(self, inputs, columns, &mut out);
        Ok(out)
    }
}

/// A one-line summary, like
/// `Symmetric { bits: 4, block: 32, len: 64, shape: None, scale: "f32", nbytes: 40, .. }`.
/// The scales and codes are left out, since one layer holds millions of them.
impl<S: Scale> core::fmt::Debug for Quantized<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let (kind, bits) = match self {
            Self::Symmetric { codes, .. } => ("Symmetric", Some(codes.bits())),
            Self::Asymmetric { codes, .. } => ("Asymmetric", Some(codes.bits())),
            Self::Adaptive { .. } => ("Adaptive", None),
        };
        let mut summary = f.debug_struct(kind);
        if let Some(bits) = bits {
            summary.field("bits", &bits);
        }
        summary
            .field("block", &self.block())
            .field("len", &self.len())
            .field("shape", &self.shape())
            .field("scale", &S::NAME)
            .field("nbytes", &self.nbytes())
            .finish_non_exhaustive()
    }
}

/// Bytes that `count` codes of `bits` each fill.
fn packed_size(count: usize, bits: u32) -> Result<usize> {
    check_bits(bits)?;
    let total_bits = count.checked_mul(bits as usize).ok_or(too_large())?;
    Ok(total_bits.div_ceil(8))
}

fn too_large() -> Error {
    malformed("len is too large to pack")
}

#[cfg(test)]
mod tests {
    use crate::{adaptive, symmetric};

    #[test]
    fn debug_prints_a_summary_instead_of_every_code() {
        let values = [0.1_f32; 64];
        let symmetric = symmetric::quantize_with::<f32>(&values, 4, 32).unwrap();
        assert_eq!(
            format!("{symmetric:?}"),
            r#"Symmetric { bits: 4, block: 32, len: 64, shape: None, scale: "f32", nbytes: 40, .. }"#
        );
        let adaptive = adaptive::quantize_with::<half::f16>(&values, 32, 0.01).unwrap();
        assert_eq!(
            format!("{adaptive:?}"),
            r#"Adaptive { block: 32, len: 64, shape: None, scale: "f16", nbytes: 26, .. }"#
        );
    }
}
