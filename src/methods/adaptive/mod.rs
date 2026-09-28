//! Mixed-precision: pick bits per block from a tolerance.

use crate::error::{check_block, Error, Result};
use crate::kernels::{min_max, quantize_asym_block};
use crate::packed::Packed;
use crate::params::{choose_bits, half_step};
use crate::scale::Scale;
use crate::tensor::Quantized;

/// Quantize with a per-block bit width chosen from `tolerance`, the largest
/// rounding error to allow for any value.
///
/// Rounding to the nearest code is off by at most half a step, so each block
/// gets the fewest bits, from 2 to 8, whose half-step is `<= tolerance`. Each
/// block is packed at its own width and concatenated.
///
/// `tolerance` is in the same units as `values`, so one number can be loose
/// for one layer and tight for the next. Pick it from the values' spread: a
/// tenth of their standard deviation gives normal weights about 5 bits a
/// block, whatever their size.
///
/// With `f16` or `bf16` scales, a value can land slightly past the tolerance,
/// since each block's scale and zero-point are rounded to fit, and several
/// times past on blocks far from zero, as [`asymmetric`](crate::asymmetric)
/// explains.
///
/// # Errors
///
/// [`Error::InvalidTolerance`] if `tolerance` is not finite and `> 0`.
/// [`Error::ToleranceTooTight`] if even 8 bits can't round a block within
/// `tolerance`, with the smallest tolerance that every block meets.
/// [`Error::InvalidBlock`] if `BLOCK == 0`.
/// [`Error::ScaleOutOfRange`] if `S` can't hold a block's scale or zero-point.
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
            columns: None,
        });
    }

    let n_blocks = values.len().div_ceil(block);
    let mut scales = Vec::with_capacity(n_blocks);
    let mut zero_points = Vec::with_capacity(n_blocks);
    let mut bits = Vec::with_capacity(n_blocks);
    let mut bytes = Vec::new();
    let mut codes = Vec::new();

    for (block_index, chunk) in values.chunks(block).enumerate() {
        let (lowest, highest) = min_max(chunk);
        let range = highest - lowest;
        let bit_width = match choose_bits(range, tolerance) {
            Some(bit_width) => bit_width,
            // An infinity decodes its block to NaN at any width, as the crate
            // docs say, so no tolerance applies to it.
            None if range.is_infinite() => 8,
            None => {
                return Err(Error::ToleranceTooTight {
                    block_index,
                    smallest_tolerance: smallest_tolerance(values, block),
                })
            }
        };
        codes.clear();
        let (scale, zero_point) =
            quantize_asym_block::<S>(chunk, block_index, bit_width, &mut codes)?;
        scales.push(scale);
        zero_points.push(zero_point);
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
        columns: None,
    })
}

/// The smallest tolerance that 8 bits meet in every block: half an 8-bit step
/// of the widest range. Infinite ranges are skipped, as in [`quantize_with`].
fn smallest_tolerance(values: &[f32], block: usize) -> f32 {
    let mut widest_range = 0.0_f32;
    for chunk in values.chunks(block) {
        let (lowest, highest) = min_max(chunk);
        let range = highest - lowest;
        if range.is_finite() {
            widest_range = widest_range.max(range);
        }
    }
    half_step(widest_range, 8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_range_uses_fewer_bits_than_wide_range() {
        let tiny = [0.500_f32, 0.501, 0.499, 0.5005];
        let wide = [0.10_f32, 0.30, 0.70, 1.10];
        let qt = quantize::<f32, 4>(&tiny, 0.002).unwrap();
        let qw = quantize::<f32, 4>(&wide, 0.002).unwrap();
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
    fn flat_block_roundtrips() {
        for value in [0.3_f32, -5.0, 1000.0] {
            let q = quantize::<f32, 32>(&[value; 32], 0.001).unwrap();
            for back in q.dequantize() {
                assert!((back - value).abs() < 1e-3, "{value} vs {back}");
            }
        }
    }

    #[test]
    fn a_zero_point_that_f16_cannot_hold_is_an_error() {
        // A tolerance this tight takes 8 bits, and 100.00 to 100.03 then needs
        // a zero-point of about -850,000, past f16's 65504.
        let values = [100.0_f32, 100.01, 100.02, 100.03];
        assert_eq!(
            quantize::<half::f16, 4>(&values, 1e-4),
            Err(Error::ScaleOutOfRange {
                block_index: 0,
                scale_type: "f16"
            })
        );
    }

    #[test]
    fn a_block_that_needs_more_than_eight_bits_is_an_error() {
        // 8 bits round block 0's range of 1 to within about 0.002, not 0.001.
        // Block 1 is twice as wide, so every block needs twice that.
        let values = [0.0_f32, 1.0, 0.0, 2.0];
        assert_eq!(
            quantize::<f32, 2>(&values, 0.001),
            Err(Error::ToleranceTooTight {
                block_index: 0,
                smallest_tolerance: half_step(2.0, 8),
            })
        );
        assert!(quantize::<f32, 2>(&values, half_step(2.0, 8)).is_ok());
    }

    #[test]
    fn an_infinity_decodes_its_block_to_nan_instead_of_missing_the_tolerance() {
        let mut values = [0.1_f32; 64];
        values[3] = f32::INFINITY;
        let back = quantize::<f32, 32>(&values, 0.001).unwrap().dequantize();
        assert!(back[..3].iter().chain(&back[4..32]).all(|v| v.is_nan()));
        assert!(back[32..].iter().all(|v| (v - 0.1).abs() < 1e-3));
    }

    #[test]
    fn rejects_non_positive_tolerance() {
        assert_eq!(
            quantize::<f32, 4>(&[1.0], 0.0).unwrap_err(),
            Error::InvalidTolerance
        );
    }
}
