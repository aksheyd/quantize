//! Mixed-precision: pick bits per block from a tolerance.

use crate::error::{check_block, Error, Result};
use crate::kernels::{min_max, quantize_asym_block};
use crate::packed::Packed;
use crate::params::choose_bits;
use crate::scale::Scale;
use crate::tensor::Quantized;

/// Quantize with a per-block bit width chosen so the half-step `<= tolerance`.
///
/// Each block is packed at its own width and concatenated.
///
/// # Errors
///
/// [`Error::InvalidTolerance`] if `tolerance` is not finite and `> 0`.
/// [`Error::InvalidBlock`] if `BLOCK == 0`.
pub fn quantize<S: Scale, const BLOCK: usize>(
    values: &[f32],
    tolerance: f32,
) -> Result<Quantized<S>> {
    quantize_with::<S>(values, BLOCK, tolerance)
}

/// Runtime-block variant of [`quantize`].
pub fn quantize_with<S: Scale>(
    values: &[f32],
    block: usize,
    tolerance: f32,
) -> Result<Quantized<S>> {
    check_block(block)?;
    if !(tolerance.is_finite() && tolerance > 0.0) {
        return Err(Error::InvalidTolerance);
    }
    if values.is_empty() {
        return Ok(Quantized::Adaptive {
            scales: Vec::new(),
            zero_points: Vec::new(),
            bytes: Vec::new(),
            bits: Vec::new(),
            block,
            len: 0,
        });
    }

    let n_blocks = values.len().div_ceil(block);
    let mut scales = Vec::with_capacity(n_blocks);
    let mut zero_points = Vec::with_capacity(n_blocks);
    let mut bits = Vec::with_capacity(n_blocks);
    let mut bytes = Vec::new();
    let mut codes = Vec::new();

    for chunk in values.chunks(block) {
        let (lowest, highest) = min_max(chunk);
        let bit_width = choose_bits(highest - lowest, tolerance);
        codes.clear();
        let (scale, zero_point) = quantize_asym_block(chunk, bit_width, &mut codes);
        scales.push(S::from_f32(scale));
        zero_points.push(S::from_f32(zero_point));
        bits.push(bit_width);
        bytes.extend_from_slice(Packed::from_i32s(&codes, bit_width).as_bytes());
    }

    Ok(Quantized::Adaptive {
        scales,
        zero_points,
        bytes,
        bits,
        block,
        len: values.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_range_uses_fewer_bits_than_wide_range() {
        let tiny = [0.500_f32, 0.501, 0.499, 0.5005];
        let wide = [0.10_f32, 0.30, 0.70, 1.10];
        let qt = quantize::<f32, 4>(&tiny, 0.001).unwrap();
        let qw = quantize::<f32, 4>(&wide, 0.001).unwrap();
        let bt = qt.block_bits().unwrap();
        let bw = qw.block_bits().unwrap();
        assert!(bt[0] < bw[0], "tiny={} wide={}", bt[0], bw[0]);
    }

    #[test]
    fn quiet_block_packs_tighter_than_eight_bit() {
        let tiny = [0.500_f32; 32];
        let q = quantize::<f32, 32>(&tiny, 0.001).unwrap();
        assert!(
            q.codes().len() < 32,
            "expected packed codes < 32 bytes, got {}",
            q.codes().len()
        );
        assert!(matches!(q, Quantized::Adaptive { .. }));
    }

    #[test]
    fn rejects_non_positive_tolerance() {
        assert_eq!(
            quantize::<f32, 4>(&[1.0], 0.0).unwrap_err(),
            Error::InvalidTolerance
        );
    }
}
