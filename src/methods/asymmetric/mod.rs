//! Asymmetric quantization: scale and zero-point per group.
//!
//! The zero-point is stored in the scale type `S`. A block far from zero,
//! compared with its range, needs a large zero-point: 4-bit values from 99.5
//! to 100.5 get -1500.5. bf16 keeps only 8 significant bits, so it rounds that
//! to -1504, and the block's worst error is 17 half-steps instead of 1. f16
//! keeps 11, but 8-bit codes push the same block's zero-point to -25500.5,
//! which f16 rounds to -25504. Use f32 scales for data far from zero.

use crate::error::{check_bits, check_block, Result};
use crate::kernels::quantize_asym_block;
use crate::packed::Packed;
use crate::scale::Scale;
use crate::tensor::Quantized;

/// Quantize into blocks of `BLOCK` using one bit width for every block.
///
/// # Errors
///
/// [`crate::Error::InvalidBits`] or [`crate::Error::InvalidBlock`], and
/// [`crate::Error::ScaleOutOfRange`] if `S` can't hold a block's scale or
/// zero-point.
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
        return Ok(Quantized::Asymmetric {
            scales: Vec::new(),
            zero_points: Vec::new(),
            codes: Packed::from_i32s(&[], bits),
            block,
            len: 0,
            columns: None,
        });
    }
    let mut scales = Vec::with_capacity(values.len().div_ceil(block));
    let mut zero_points = Vec::with_capacity(values.len().div_ceil(block));
    let mut codes = Vec::with_capacity(values.len());
    for (block_index, chunk) in values.chunks(block).enumerate() {
        let (scale, zero_point) = quantize_asym_block::<S>(chunk, block_index, bits, &mut codes)?;
        scales.push(scale);
        zero_points.push(zero_point);
    }
    Ok(Quantized::Asymmetric {
        scales,
        zero_points,
        codes: Packed::from_i32s(&codes, bits),
        block,
        len: values.len(),
        columns: None,
    })
}

/// Quantize the entire tensor with one scale and one zero-point.
pub fn quantize_tensor<S: Scale, const BITS: u32>(values: &[f32]) -> Result<Quantized<S>> {
    quantize_with::<S>(values, BITS, values.len().max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_block_uses_full_grid() {
        let w = [0.10_f32, 0.30, 0.70, 1.10];
        let q = quantize::<f32, 8, 4>(&w).unwrap();
        assert!(matches!(q, Quantized::Asymmetric { .. }));
        assert!(!q.zero_points().is_empty());
        let back = q.dequantize();
        for (a, b) in w.iter().zip(&back) {
            assert!((a - b).abs() < 0.02, "{a} vs {b}");
        }
    }

    #[test]
    fn codes_are_picked_against_the_stored_scale_and_zero_point() {
        // All positive, so each zero-point sits near -150, where bf16 stores
        // only whole numbers. Codes picked against the stored zero-point take
        // up that rounding, so the error stays close to f32's.
        let w: Vec<f32> = (0..1024).map(|i| 0.6 + (i as f32).sin() * 0.5).collect();
        let squared_error =
            |back: Vec<f32>| -> f32 { w.iter().zip(back).map(|(a, b)| (a - b) * (a - b)).sum() };
        let exact = squared_error(quantize_with::<f32>(&w, 8, 32).unwrap().dequantize());
        let rounded = squared_error(quantize_with::<half::bf16>(&w, 8, 32).unwrap().dequantize());
        assert!(rounded < exact * 1.2, "{rounded} vs {exact}");
    }

    #[test]
    fn a_zero_point_that_f16_cannot_hold_is_an_error() {
        // At 16 bits, 0.02 to 0.05 needs a zero-point of about -76,000, past
        // f16's 65504. Stored anyway, it would decode every value to infinity.
        let values = [0.02_f32, 0.03, 0.04, 0.05];
        assert_eq!(
            quantize_with::<half::f16>(&values, 16, 4),
            Err(crate::Error::ScaleOutOfRange {
                block_index: 0,
                scale_type: "f16"
            })
        );
        let back = quantize_with::<f32>(&values, 16, 4).unwrap().dequantize();
        for (a, b) in values.iter().zip(&back) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn flat_block_roundtrips() {
        for value in [0.3_f32, -5.0, 1000.0] {
            let q = quantize::<f32, 4, 32>(&[value; 32]).unwrap();
            for back in q.dequantize() {
                assert!((back - value).abs() < 1e-3, "{value} vs {back}");
            }
        }
    }

    #[test]
    fn nan_only_block_decodes_to_zero() {
        let q = quantize::<f32, 8, 4>(&[f32::NAN; 4]).unwrap();
        assert_eq!(q.dequantize(), [0.0; 4]);
    }
}
