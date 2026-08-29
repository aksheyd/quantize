//! Runtime scheme selection — one entry point, scheme-specific I/O inside.

use crate::error::Result;
use crate::scale::Scale;
use crate::tensor::Quantized;
use crate::{adaptive, asymmetric, symmetric};

/// Which algorithm to run. Pick this from config or a CLI flag; the returned
/// [`Quantized`] variant is the scheme that ran.
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
        /// Maximum half-step of the integer grid.
        tolerance: f32,
    },
}

impl Scheme {
    /// Symmetric 8-bit blocks of 32 — the common default.
    pub const Q8_32: Self = Self::Symmetric { bits: 8, block: 32 };

    /// Symmetric 4-bit blocks of 32 — GGML Q4_0-shaped storage.
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
}
