//! Recoverable failures from quantization and dequantization.

use core::fmt;

/// An error produced by a fallible quantization API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// `bits` is outside the supported `2..=16` range.
    InvalidBits {
        /// Requested bit width.
        bits: u32,
    },
    /// Block length must be at least 1.
    InvalidBlock {
        /// Requested block length.
        block: usize,
    },
    /// Reconstruction tolerance must be finite and strictly positive.
    InvalidTolerance,
    /// A block's scale or zero-point is too large for the scale type. f16, for
    /// example, holds magnitudes up to 65504.
    ScaleOutOfRange {
        /// Index of the first block that doesn't fit.
        block_index: usize,
        /// The scale type's [`NAME`](crate::Scale::NAME), like `f16`.
        scale_type: &'static str,
    },
    /// A buffer holds a different number of values than the quantized tensor.
    LengthMismatch {
        /// Length required by the quantized tensor.
        expected: usize,
        /// Length the caller actually passed.
        got: usize,
    },
    /// `columns` is zero or doesn't split `len` values into whole rows.
    ShapeMismatch {
        /// Number of values that were to be split into rows.
        len: usize,
        /// Requested row length.
        columns: usize,
    },
    /// [`set_shape`](crate::Quantized::set_shape) got a `rows × columns`
    /// shape that doesn't hold the tensor's `len` values.
    MatrixMismatch {
        /// Requested number of rows.
        rows: usize,
        /// Requested number of columns.
        columns: usize,
        /// Number of values in the tensor.
        len: usize,
    },
    /// [`matmul`](crate::Quantized::matmul) and
    /// [`dequantize_row`](crate::Quantized::dequantize_row) need a matrix, but
    /// the tensor is a flat vector.
    NotAMatrix {
        /// Number of values in the vector.
        len: usize,
    },
    /// [`dequantize_row`](crate::Quantized::dequantize_row) asked for a row
    /// past the end of the matrix.
    RowOutOfRange {
        /// Requested row.
        row: usize,
        /// Number of rows in the matrix.
        rows: usize,
    },
    /// A `batch × rows` matmul result has more values than can be allocated.
    OutputTooLarge {
        /// Number of input vectors.
        batch: usize,
        /// Number of matrix rows.
        rows: usize,
    },
    /// Saved bytes, or a tensor built by hand, don't hold a valid tensor.
    Malformed {
        /// What is wrong.
        reason: &'static str,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBits { bits } => {
                write!(f, "bit width {bits} is outside the supported range 2..=16")
            }
            Self::InvalidBlock { block } => {
                write!(f, "block size {block} must be at least 1")
            }
            Self::InvalidTolerance => {
                write!(f, "tolerance must be a finite number greater than 0")
            }
            Self::ScaleOutOfRange {
                block_index,
                scale_type,
            } => {
                write!(
                    f,
                    "block {block_index}'s scale or zero-point doesn't fit in {scale_type}; use f32 scales"
                )
            }
            Self::LengthMismatch { expected, got } => {
                write!(f, "length mismatch: expected {expected}, got {got}")
            }
            Self::ShapeMismatch { len, columns } => {
                write!(
                    f,
                    "{len} values can't be split into rows of {columns} columns"
                )
            }
            Self::MatrixMismatch { rows, columns, len } => match rows.checked_mul(*columns) {
                Some(size) => write!(
                    f,
                    "a {rows} x {columns} matrix holds {size} values, but the tensor has {len}"
                ),
                None => write!(
                    f,
                    "a {rows} x {columns} matrix holds too many values, but the tensor has {len}"
                ),
            },
            Self::NotAMatrix { len } => {
                write!(
                    f,
                    "this tensor is a flat vector of {len} values, not a matrix; call set_shape(rows, columns) first"
                )
            }
            Self::RowOutOfRange { row, rows } => {
                write!(f, "row {row} is past the end of a matrix with {rows} rows")
            }
            Self::OutputTooLarge { batch, rows } => {
                write!(f, "a {batch} x {rows} output is too large to allocate")
            }
            Self::Malformed { reason } => write!(f, "malformed tensor: {reason}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// Result alias for this crate.
pub type Result<T, E = Error> = core::result::Result<T, E>;

pub(crate) fn check_bits(bits: u32) -> Result<()> {
    if (2..=16).contains(&bits) {
        Ok(())
    } else {
        Err(Error::InvalidBits { bits })
    }
}

pub(crate) fn check_block(block: usize) -> Result<()> {
    if block == 0 {
        Err(Error::InvalidBlock { block })
    } else {
        Ok(())
    }
}

pub(crate) fn check_len(expected: usize, got: usize) -> Result<()> {
    if expected == got {
        Ok(())
    } else {
        Err(Error::LengthMismatch { expected, got })
    }
}

pub(crate) fn malformed(reason: &'static str) -> Error {
    Error::Malformed { reason }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_bits_rejects_one_and_seventeen() {
        assert!(matches!(check_bits(1), Err(Error::InvalidBits { bits: 1 })));
        assert!(matches!(
            check_bits(17),
            Err(Error::InvalidBits { bits: 17 })
        ));
    }

    #[test]
    fn display_says_how_many_values_the_matrix_holds() {
        let err = Error::MatrixMismatch {
            rows: 3,
            columns: 32,
            len: 64,
        };
        assert_eq!(
            err.to_string(),
            "a 3 x 32 matrix holds 96 values, but the tensor has 64"
        );
        let err = Error::MatrixMismatch {
            rows: usize::MAX,
            columns: 2,
            len: 64,
        };
        assert_eq!(
            err.to_string(),
            format!(
                "a {} x 2 matrix holds too many values, but the tensor has 64",
                usize::MAX
            )
        );
    }

    #[test]
    fn display_mentions_expected_length() {
        let err = Error::LengthMismatch {
            expected: 4,
            got: 1,
        };
        assert_eq!(err.to_string(), "length mismatch: expected 4, got 1");
    }

    #[test]
    fn display_suggests_f32_scales() {
        let err = Error::ScaleOutOfRange {
            block_index: 3,
            scale_type: "f16",
        };
        assert_eq!(
            err.to_string(),
            "block 3's scale or zero-point doesn't fit in f16; use f32 scales"
        );
    }

    #[test]
    fn display_names_the_values_and_columns_that_do_not_fit() {
        let err = Error::ShapeMismatch {
            len: 64,
            columns: 24,
        };
        assert_eq!(
            err.to_string(),
            "64 values can't be split into rows of 24 columns"
        );
    }
}
