//! One enum, one variant per scheme.

use crate::decode::{dequant_adaptive, dequant_asym, dequant_sym, dot_of};
use crate::error::{check_len, Result};
use crate::packed::Packed;
use crate::scale::Scale;

/// Packed codes and the scheme that produced them.
#[derive(Clone, Debug)]
pub enum Quantized<S: Scale> {
    /// One scale per block.
    Symmetric {
        scales: Vec<S>,
        codes: Packed,
        block: usize,
        len: usize,
    },
    /// Scale and zero-point per block.
    Asymmetric {
        scales: Vec<S>,
        zero_points: Vec<S>,
        codes: Packed,
        block: usize,
        len: usize,
    },
    /// Per-block bit width; codes packed at that width.
    Adaptive {
        scales: Vec<S>,
        zero_points: Vec<S>,
        bytes: Vec<u8>,
        bits: Vec<u32>,
        block: usize,
        len: usize,
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

    pub fn block_bits(&self) -> Option<&[u32]> {
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
            } => dequant_adaptive(scales, zero_points, bytes, bits, *block, *len, out),
        }
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
}
