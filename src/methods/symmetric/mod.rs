//! Symmetric quantization: one scale per group, codes centered on zero.

use crate::error::{check_bits, check_block, Result};
use crate::kernels::quantize_sym_packed;
use crate::packed::Packed;
use crate::scale::Scale;
use crate::tensor::Quantized;

/// Quantize `values` into fixed-size blocks of `BLOCK` with `BITS`-wide codes.
///
/// # Errors
///
/// [`crate::Error::InvalidBits`] or [`crate::Error::InvalidBlock`].
pub fn quantize<S: Scale, const BITS: u32, const BLOCK: usize>(
    values: &[f32],
) -> Result<Quantized<S>> {
    quantize_with::<S>(values, BITS, BLOCK)
}

/// Runtime-width variant of [`quantize`].
pub fn quantize_with<S: Scale>(values: &[f32], bits: u32, block: usize) -> Result<Quantized<S>> {
    check_bits(bits)?;
    check_block(block)?;
    if values.is_empty() {
        return Ok(Quantized::Symmetric {
            scales: Vec::new(),
            codes: Packed::from_raw(Vec::new(), bits, 0),
            block,
            len: 0,
        });
    }
    let (scales_f, codes) = quantize_sym_packed(values, bits, block);
    Ok(Quantized::Symmetric {
        scales: scales_f.into_iter().map(S::from_f32).collect(),
        codes,
        block,
        len: values.len(),
    })
}

/// Quantize the entire tensor with a single scale.
pub fn quantize_tensor<S: Scale, const BITS: u32>(values: &[f32]) -> Result<Quantized<S>> {
    quantize_with::<S>(values, BITS, values.len().max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eight_bit_roundtrip_stays_within_half_step() {
        let w = [0.42_f32, -0.10, 0.70, -0.50];
        let q = quantize::<f32, 8, 4>(&w).unwrap();
        let back = q.dequantize();
        for (a, b) in w.iter().zip(&back) {
            assert!((a - b).abs() < 0.01, "{a} vs {b}");
        }
    }

    #[test]
    fn packed_four_bit_uses_half_byte_per_code() {
        let w = [0.1_f32; 32];
        let q = quantize::<f32, 4, 32>(&w).unwrap();
        assert_eq!(q.codes().len(), 16);
    }

    #[test]
    fn remainder_block_roundtrips() {
        let w: Vec<f32> = (0..40).map(|i| (i as f32) * 0.01 - 0.2).collect();
        let q = quantize::<f32, 8, 32>(&w).unwrap();
        assert_eq!(q.len(), 40);
        let back = q.dequantize();
        for (a, b) in w.iter().zip(&back) {
            assert!((a - b).abs() < 0.01, "{a} vs {b}");
        }
    }

    #[test]
    fn dequantize_into_rejects_wrong_length() {
        let w = [0.1_f32; 8];
        let q = quantize::<f32, 8, 8>(&w).unwrap();
        let mut out = [0.0f32; 3];
        assert!(matches!(
            q.dequantize_into(&mut out),
            Err(crate::Error::LengthMismatch {
                expected: 8,
                got: 3
            })
        ));
    }

    #[test]
    fn four_bit_remainder_roundtrips() {
        let w: Vec<f32> = (0..40).map(|i| (i as f32) * 0.02 - 0.4).collect();
        let q = quantize::<f32, 4, 32>(&w).unwrap();
        let back = q.dequantize();
        for (a, b) in w.iter().zip(&back) {
            assert!((a - b).abs() < 0.08, "{a} vs {b}");
        }
    }

    #[test]
    fn fused_dot_matches_dequant_then_dot() {
        let w: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01 - 0.3).collect();
        let q = quantize::<f32, 8, 32>(&w).unwrap();
        let recon = q.dequantize();
        let naive: f32 = recon.iter().zip(&w).map(|(a, b)| a * b).sum();
        let fused = q.dot(&w).unwrap();
        assert!((naive - fused).abs() < 1e-4, "{naive} vs {fused}");
    }

    #[test]
    fn fused_dot_four_bit_matches_dequant_then_dot() {
        let w: Vec<f32> = (0..64).map(|i| (i as f32) * 0.02 - 0.4).collect();
        let q = quantize::<f32, 4, 32>(&w).unwrap();
        let recon = q.dequantize();
        let naive: f32 = recon.iter().zip(&w).map(|(a, b)| a * b).sum();
        let fused = q.dot(&w).unwrap();
        assert!((naive - fused).abs() < 1e-4, "{naive} vs {fused}");
    }

    #[test]
    fn fused_matmul_matches_dequant_then_multiply() {
        let w: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01 - 0.3).collect();
        let q = quantize::<f32, 8, 32>(&w).unwrap();
        let recon = q.dequantize();
        let rhs: Vec<f32> = (0..32).map(|i| (i as f32) * 0.02 - 0.1).collect();
        let fused = q.matmul(&rhs, 32).unwrap();
        for row in 0..2 {
            let naive: f32 = recon[row * 32..(row + 1) * 32]
                .iter()
                .zip(&rhs)
                .map(|(a, b)| a * b)
                .sum();
            assert!(
                (naive - fused[row]).abs() < 1e-4,
                "{naive} vs {}",
                fused[row]
            );
        }
    }

    #[test]
    fn fused_matmul_four_bit_matches_dequant_then_multiply() {
        let w: Vec<f32> = (0..64).map(|i| (i as f32) * 0.02 - 0.4).collect();
        let q = quantize::<f32, 4, 32>(&w).unwrap();
        let recon = q.dequantize();
        let rhs: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03 - 0.2).collect();
        let fused = q.matmul(&rhs, 32).unwrap();
        for row in 0..2 {
            let naive: f32 = recon[row * 32..(row + 1) * 32]
                .iter()
                .zip(&rhs)
                .map(|(a, b)| a * b)
                .sum();
            assert!(
                (naive - fused[row]).abs() < 1e-4,
                "{naive} vs {}",
                fused[row]
            );
        }
    }

    #[test]
    fn matmul_batch_is_row_major() {
        let w: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01 - 0.3).collect();
        let q = quantize::<f32, 8, 32>(&w).unwrap();
        let recon = q.dequantize();
        let rhs: Vec<f32> = (0..64).map(|i| (i as f32) * 0.02 - 0.15).collect();
        let fused = q.matmul(&rhs, 32).unwrap();
        assert_eq!(fused.len(), 4);
        for vector in 0..2 {
            for row in 0..2 {
                let naive: f32 = recon[row * 32..(row + 1) * 32]
                    .iter()
                    .zip(&rhs[vector * 32..(vector + 1) * 32])
                    .map(|(a, b)| a * b)
                    .sum();
                let got = fused[vector * 2 + row];
                assert!((naive - got).abs() < 1e-4, "{naive} vs {got}");
            }
        }
    }

    #[test]
    fn matmul_rejects_zero_columns() {
        let w = [0.1_f32; 8];
        let q = quantize::<f32, 8, 8>(&w).unwrap();
        assert!(matches!(
            q.matmul(&w, 0),
            Err(crate::Error::InvalidBlock { block: 0 })
        ));
    }

    #[test]
    fn matmul_rejects_indivisible_columns() {
        let w: Vec<f32> = (0..40).map(|i| (i as f32) * 0.01).collect();
        let q = quantize::<f32, 8, 32>(&w).unwrap();
        let rhs = [0.0_f32; 32];
        assert!(matches!(
            q.matmul(&rhs, 32),
            Err(crate::Error::LengthMismatch {
                expected: 40,
                got: 32
            })
        ));
    }
}
