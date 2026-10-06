//! Runtime scheme selection — one entry point, scheme-specific I/O inside.

use core::fmt;
use core::str::FromStr;

use crate::error::{Error, Result};
use crate::scale::Scale;
use crate::tensor::Quantized;
use crate::{adaptive, asymmetric, symmetric};

/// Which algorithm to run. Pick this from config or a CLI flag: it prints as
/// text like `symmetric(bits=4, block=32)`, and [`parse`](str::parse) reads
/// that text back, or the name of a constant, `Q8_32` or `Q4_32`:
///
/// ```
/// use quantize::Scheme;
///
/// let scheme: Scheme = "adaptive(block=32, tolerance=0.002)".parse().unwrap();
/// assert_eq!(scheme, Scheme::Adaptive { block: 32, tolerance: 0.002 });
/// assert_eq!(scheme.to_string(), "adaptive(block=32, tolerance=0.002)");
/// assert_eq!("Q4_32".parse(), Ok(Scheme::Q4_32));
/// ```
///
/// `Scheme` doesn't implement serde's traits. In a serde config, read it as a
/// string and parse it, for example with `serde_with`'s
/// `#[serde_as(as = "DisplayFromStr")]`.
///
/// The [`Quantized`] variant that [`quantize`](Self::quantize) returns is the
/// scheme that ran.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scheme {
    /// Symmetric block quantization.
    Symmetric {
        /// Integer width, `2..=16`.
        bits: u32,
        /// Elements per scale.
        block: usize,
    },
    /// Asymmetric block quantization.
    Asymmetric {
        /// Integer width, `2..=16`.
        bits: u32,
        /// Elements per scale.
        block: usize,
    },
    /// Per-block bit width from a reconstruction tolerance.
    Adaptive {
        /// Elements per scale / bit-width decision.
        block: usize,
        /// Largest rounding error to allow for any value, in the values' own
        /// units, as in [`adaptive::quantize`].
        tolerance: f32,
    },
}

impl Scheme {
    /// Symmetric 8-bit blocks of 32 — the common default.
    pub const Q8_32: Self = Self::Symmetric { bits: 8, block: 32 };

    /// Symmetric 4-bit blocks of 32. With `f16` scales it is the same size as
    /// GGML Q4_0 (4.5 bits per value), but the bytes are laid out differently.
    pub const Q4_32: Self = Self::Symmetric { bits: 4, block: 32 };

    /// Run this scheme on `values`.
    pub fn quantize<S: Scale>(self, values: &[f32]) -> Result<Quantized<S>> {
        match self {
            Self::Symmetric { bits, block } => symmetric::quantize_with::<S>(values, bits, block),
            Self::Asymmetric { bits, block } => asymmetric::quantize_with::<S>(values, bits, block),
            Self::Adaptive { block, tolerance } => {
                adaptive::quantize_with::<S>(values, block, tolerance)
            }
        }
    }
}

/// Prints `symmetric(bits=4, block=32)`, `asymmetric(bits=8, block=32)`, or
/// `adaptive(block=32, tolerance=0.002)`, which [`parse`](str::parse) reads
/// back.
impl fmt::Display for Scheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Symmetric { bits, block } => write!(f, "symmetric(bits={bits}, block={block})"),
            Self::Asymmetric { bits, block } => {
                write!(f, "asymmetric(bits={bits}, block={block})")
            }
            Self::Adaptive { block, tolerance } => {
                write!(f, "adaptive(block={block}, tolerance={tolerance})")
            }
        }
    }
}

/// Reads what [`Display`](fmt::Display) prints, with or without the spaces,
/// or the name `Q8_32` or `Q4_32`.
impl FromStr for Scheme {
    type Err = Error;

    fn from_str(text: &str) -> Result<Self> {
        parse_scheme(text.trim()).ok_or_else(|| Error::InvalidScheme {
            text: text.to_string(),
        })
    }
}

/// The scheme `text` is written as, or `None` if it isn't one.
fn parse_scheme(text: &str) -> Option<Scheme> {
    match text {
        "Q8_32" => return Some(Scheme::Q8_32),
        "Q4_32" => return Some(Scheme::Q4_32),
        _ => {}
    }
    // "symmetric(bits=4, block=32)" is the kind "symmetric", then the
    // arguments ("bits", "4") and ("block", "32").
    let (kind, arguments) = text.strip_suffix(')')?.split_once('(')?;
    let arguments: Vec<(&str, &str)> = arguments
        .split(',')
        .map(|argument| {
            let (name, value) = argument.split_once('=')?;
            Some((name.trim(), value.trim()))
        })
        .collect::<Option<_>>()?;
    Some(match (kind.trim(), arguments.as_slice()) {
        ("symmetric", [("bits", bits), ("block", block)]) => Scheme::Symmetric {
            bits: bits.parse().ok()?,
            block: block.parse().ok()?,
        },
        ("asymmetric", [("bits", bits), ("block", block)]) => Scheme::Asymmetric {
            bits: bits.parse().ok()?,
            block: block.parse().ok()?,
        },
        ("adaptive", [("block", block), ("tolerance", tolerance)]) => Scheme::Adaptive {
            block: block.parse().ok()?,
            tolerance: tolerance.parse().ok()?,
        },
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheme_enum_matches_direct_call() {
        let w = [0.42_f32, -0.10, 0.70, -0.50];
        let via = Scheme::Symmetric { bits: 8, block: 4 }
            .quantize::<f32>(&w)
            .unwrap();
        let direct = symmetric::quantize::<f32, 8, 4>(&w).unwrap();
        assert_eq!(via.dequantize(), direct.dequantize());
    }

    #[test]
    fn every_scheme_reads_back_what_it_prints() {
        assert_eq!(Scheme::Q4_32.to_string(), "symmetric(bits=4, block=32)");
        let schemes = [
            Scheme::Q8_32,
            Scheme::Asymmetric { bits: 3, block: 7 },
            Scheme::Adaptive {
                block: 32,
                tolerance: 0.1 * 0.0173,
            },
        ];
        for scheme in schemes {
            assert_eq!(scheme.to_string().parse(), Ok(scheme), "{scheme}");
        }
    }

    #[test]
    fn parse_reads_the_constants_and_text_without_spaces() {
        assert_eq!("Q8_32".parse(), Ok(Scheme::Q8_32));
        assert_eq!(
            "asymmetric(bits=8,block=64)".parse(),
            Ok(Scheme::Asymmetric { bits: 8, block: 64 })
        );
    }

    #[test]
    fn parse_rejects_text_that_is_not_a_scheme() {
        for text in [
            "",
            "q4_32",
            "symetric(bits=4, block=32)",
            "symmetric(bits=4)",
            "symmetric(block=32, bits=4)",
            "adaptive(bits=4, block=32)",
            "symmetric(bits=four, block=32)",
            "symmetric:4:32",
        ] {
            assert_eq!(
                text.parse::<Scheme>(),
                Err(Error::InvalidScheme {
                    text: text.to_string()
                }),
            );
        }
    }
}
