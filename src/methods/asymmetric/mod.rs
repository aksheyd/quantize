//! Asymmetric quantization: scale and zero-point per group.

use crate::error::{check_bits, check_block, Result};
use crate::kernels::quantize_asym_block;
use crate::packed::Packed;
use crate::scale::Scale;
use crate::tensor::Quantized;

/// Quantize into blocks of `BLOCK` using one bit width for every block.
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
        });
    }
    let mut scales = Vec::with_capacity(values.len().div_ceil(block));
    let mut zero_points = Vec::with_capacity(values.len().div_ceil(block));
    let mut codes = Vec::with_capacity(values.len());
    for chunk in values.chunks(block) {
        let (s, z) = quantize_asym_block(chunk, bits, &mut codes);
        scales.push(S::from_f32(s));
        zero_points.push(S::from_f32(z));
    }
    Ok(Quantized::Asymmetric {
        scales,
        zero_points,
        codes: Packed::from_i32s(&codes, bits),
        block,
        len: values.len(),
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
}
