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
    /// Output or partner buffer length does not match the quantized tensor.
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
    /// A `batch × rows` matmul result has more values than can be allocated.
    OutputTooLarge {
        /// Number of input vectors.
        batch: usize,
        /// Number of matrix rows.
        rows: usize,
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
            Self::LengthMismatch { expected, got } => {
                write!(f, "length mismatch: expected {expected}, got {got}")
            }
            Self::ShapeMismatch { len, columns } => {
                write!(
                    f,
                    "{len} values can't be split into rows of {columns} columns"
                )
            }
            Self::OutputTooLarge { batch, rows } => {
                write!(f, "a {batch} x {rows} output is too large to allocate")
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
    fn display_mentions_expected_length() {
        let err = Error::LengthMismatch {
            expected: 4,
            got: 1,
        };
        assert_eq!(err.to_string(), "length mismatch: expected 4, got 1");
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
