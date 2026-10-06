//! Recoverable failures from quantization and dequantization.

use core::fmt;

/// An error produced by a fallible quantization API.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
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
    /// Even 8 bits, the most [`adaptive`](crate::adaptive) gives a block,
    /// can't round block `block_index` within the tolerance.
    ToleranceTooTight {
        /// Index of the first block that misses the tolerance.
        block_index: usize,
        /// The smallest tolerance that 8 bits meet in every block.
        smallest_tolerance: f32,
    },
    /// A block's scale or zero-point is too large for the scale type. f16, for
    /// example, holds magnitudes up to 65504.
    ScaleOutOfRange {
        /// Index of the first block that doesn't fit.
        block_index: usize,
        /// The scale type's [`NAME`](crate::Scale::NAME), like `f16`.
        scale_type: &'static str,
    },
    /// A buffer holds a different number of values than the call needs.
    LengthMismatch {
        /// Length the call needs, like the tensor's length.
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
    /// [`matmul`](crate::Quantized::matmul),
    /// [`matmul_into`](crate::Quantized::matmul_into), and
    /// [`dequantize_row_into`](crate::Quantized::dequantize_row_into) need a
    /// matrix, but the tensor is a flat vector.
    NotAMatrix {
        /// Number of values in the vector.
        len: usize,
    },
    /// [`dequantize_row_into`](crate::Quantized::dequantize_row_into) asked
    /// for a row past the end of the matrix.
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
    /// [`from_bytes`](crate::Quantized::from_bytes) read a tensor saved with
    /// another scale type.
    ScaleMismatch {
        /// The scale type the tensor was saved with, as its bytes name it.
        saved: String,
        /// The scale type it was loaded as.
        expected: &'static str,
    },
    /// [`from_bytes`](crate::Quantized::from_bytes) read a tensor saved in a
    /// newer format version than this version of quantize reads.
    NewerFormat {
        /// The format version the tensor was saved in.
        version: u8,
    },
    /// [`Scheme`](crate::Scheme)'s [`parse`](str::parse) read text that isn't
    /// written like a scheme.
    InvalidScheme {
        /// The text that was read.
        text: String,
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
            Self::ToleranceTooTight {
                block_index,
                smallest_tolerance,
            } => {
                write!(
                    f,
                    "8 bits can't round block {block_index} within the tolerance; use a tolerance of at least {smallest_tolerance}"
                )
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
            Self::ScaleMismatch { saved, expected } => {
                write!(
                    f,
                    "the tensor was saved with {saved} scales, not {expected}"
                )
            }
            Self::NewerFormat { version } => {
                write!(
                    f,
                    "the tensor was saved by a newer version of quantize, in format version {version}; upgrade quantize to load it"
                )
            }
            Self::InvalidScheme { text } => {
                write!(
                    f,
                    "{text:?} isn't a scheme; write one like symmetric(bits=4, block=32), asymmetric(bits=8, block=32), or adaptive(block=32, tolerance=0.002)"
                )
            }
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
    fn display_suggests_a_tolerance_every_block_meets() {
        let err = Error::ToleranceTooTight {
            block_index: 3,
            smallest_tolerance: 0.25,
        };
        assert_eq!(
            err.to_string(),
            "8 bits can't round block 3 within the tolerance; use a tolerance of at least 0.25"
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

    #[test]
    fn display_names_both_scale_types() {
        let err = Error::ScaleMismatch {
            saved: "f16".to_string(),
            expected: "f32",
        };
        assert_eq!(
            err.to_string(),
            "the tensor was saved with f16 scales, not f32"
        );
    }

    #[test]
    fn display_suggests_a_newer_quantize() {
        let err = Error::NewerFormat { version: 2 };
        assert_eq!(
            err.to_string(),
            "the tensor was saved by a newer version of quantize, in format version 2; upgrade quantize to load it"
        );
    }

    #[test]
    fn display_shows_how_to_write_a_scheme() {
        let err = Error::InvalidScheme {
            text: "symmetric:4:32".to_string(),
        };
        assert_eq!(
            err.to_string(),
            r#""symmetric:4:32" isn't a scheme; write one like symmetric(bits=4, block=32), asymmetric(bits=8, block=32), or adaptive(block=32, tolerance=0.002)"#
        );
    }
}
