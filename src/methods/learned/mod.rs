//! After codes are chosen, pick a better scale and zero-point.
//!
//! We want: `original ≈ scale * (code - zero_point)`.
//! Same as: `original ≈ scale * code + offset`, with
//! `offset = -scale * zero_point`. Codes stay fixed.
//!
//! A symmetric block has no zero-point, so its line must pass through zero:
//! `original ≈ scale * code`, and only the scale is fitted.
//!
//! Once the line moves, a value may sit closer to a neighboring code.
//! [`alternate`] rounds every value again on the new line and refits, until no
//! code moves.

use crate::decode::unpack_codes;
use crate::error::{Result, check_len};
use crate::packed::Packed;
use crate::params::{largest_code, smallest_code};
use crate::scale::Scale;
use crate::tensor::Quantized;

/// Best-fit `scale` and `zero_point` for `values ≈ scale * (codes - zero_point)`.
///
/// When every code is the same there is no slope to fit, so the scale stays
/// `1.0` and only the offset is fitted: every code decodes to the mean value.
///
/// When every value is the same but the codes differ, the best line is flat,
/// and `scale * (codes - zero_point)` is flat only at 0. So the zero-point
/// stays 0 and only the scale is fitted, giving the line through zero that
/// comes closest to the values.
pub fn fit_scale_and_zero_point(values: &[f32], codes: &[i32]) -> (f32, f32) {
    debug_assert_eq!(values.len(), codes.len());
    let count = values.len() as f64;
    if count == 0.0 {
        return (1.0, 0.0);
    }

    // Means first, then spreads around the means, in f64. Summing raw products
    // in one pass instead cancels out on values far from zero.
    let mean_code = codes.iter().map(|&code| code as f64).sum::<f64>() / count;
    let mean_value = values.iter().map(|&value| value as f64).sum::<f64>() / count;
    let mut code_spread = 0.0;
    let mut code_value_spread = 0.0;
    for (&value, &code) in values.iter().zip(codes) {
        let centered_code = code as f64 - mean_code;
        let centered_value = value as f64 - mean_value;
        code_spread += centered_code * centered_code;
        code_value_spread += centered_code * centered_value;
    }
    if code_spread == 0.0 {
        return (1.0, (mean_code - mean_value) as f32);
    }

    // Line of best fit: value ≈ scale * code + offset.
    let scale = code_value_spread / code_spread;
    let offset = mean_value - scale * mean_code;
    if scale.abs() < 1e-12 {
        return (fit_scale(values, codes), 0.0);
    }
    let zero_point = -offset / scale;
    (scale as f32, zero_point as f32)
}

/// Best-fit `scale` for `values ≈ scale * codes`, the line through zero.
///
/// When every code is 0 there is no slope to fit, and any scale decodes the
/// block to 0, so the scale stays `1.0`.
fn fit_scale(values: &[f32], codes: &[i32]) -> f32 {
    let mut sum_code_squared = 0.0;
    let mut sum_code_times_value = 0.0;
    for (&value, &code) in values.iter().zip(codes) {
        sum_code_squared += code as f64 * code as f64;
        sum_code_times_value += code as f64 * value as f64;
    }
    if sum_code_squared == 0.0 {
        return 1.0;
    }
    (sum_code_times_value / sum_code_squared) as f32
}

/// Recompute each block's scale, and its zero-point if it has one. Codes do
/// not change.
///
/// A symmetric tensor stays symmetric: only its scales are fitted, so it keeps
/// its size and its faster decoding. A block keeps its old parameters unless
/// the new ones lower its mean squared error. Empty input is left as-is.
///
/// A block's worst error can still rise, so a value in an adaptive tensor can
/// land past the tolerance it was quantized with.
///
/// # Errors
///
/// [`crate::Error::LengthMismatch`] if `values.len() != quantized.len()`.
pub fn refine<S: Scale>(quantized: &mut Quantized<S>, values: &[f32]) -> Result<()> {
    check_len(quantized.len(), values.len())?;
    if quantized.is_empty() {
        return Ok(());
    }
    let block = quantized.block();
    let mut codes = vec![0i32; quantized.len()];
    unpack_codes(quantized, &mut codes);
    let blocks = values.chunks(block).zip(codes.chunks(block));

    // After rounding to `S`, a fit can decode worse than the old parameters.
    match quantized {
        Quantized::Symmetric { scales, .. } => {
            let zero = S::from_f32(0.0);
            for ((block_values, block_codes), scale) in blocks.zip(scales) {
                let fitted = S::from_f32(fit_scale(block_values, block_codes));
                let error = |scale| squared_error(block_values, block_codes, (scale, zero));
                if error(fitted) < error(*scale) {
                    *scale = fitted;
                }
            }
        }
        Quantized::Asymmetric {
            scales,
            zero_points,
            ..
        }
        | Quantized::Adaptive {
            scales,
            zero_points,
            ..
        } => {
            let parameters = scales.iter_mut().zip(zero_points);
            for ((block_values, block_codes), (scale, zero_point)) in blocks.zip(parameters) {
                let (fitted_scale, fitted_zero_point) =
                    fit_scale_and_zero_point(block_values, block_codes);
                let fitted = (S::from_f32(fitted_scale), S::from_f32(fitted_zero_point));
                let error = |parameters| squared_error(block_values, block_codes, parameters);
                if error(fitted) < error((*scale, *zero_point)) {
                    (*scale, *zero_point) = fitted;
                }
            }
        }
    }
    Ok(())
}

/// Sum of squared differences between `values` and the decoded `codes`.
fn squared_error<S: Scale>(values: &[f32], codes: &[i32], (scale, zero_point): (S, S)) -> f64 {
    let (scale, zero_point) = (scale.to_f32(), zero_point.to_f32());
    values
        .iter()
        .zip(codes)
        .map(|(&value, &code)| {
            let error = (value - (code as f32 - zero_point) * scale) as f64;
            error * error
        })
        .sum()
}

/// Let the codes move too: [`refine`], then round every value to its nearest
/// code on its block's new line, and repeat until no code moves.
///
/// Rounding picks the best codes for each line, and [`refine`] keeps a line
/// only if it lowers its block's mean squared error, so neither step raises
/// that error: it ends no higher than [`refine`] alone leaves it. The tensor
/// keeps its scheme and each block its bit width, so its size doesn't change.
///
/// A block's worst error can still rise, so a value in an adaptive tensor can
/// land past the tolerance it was quantized with.
///
/// Blocks of 32 settle within about 15 passes, but a block as large as a whole
/// tensor can keep moving codes for thousands, so this stops after 100. It
/// returns `true` once no code moves, or `false` if it stopped first: call it
/// again while it returns `false`.
///
/// # Errors
///
/// [`crate::Error::LengthMismatch`] if `values.len() != quantized.len()`.
pub fn alternate<S: Scale>(quantized: &mut Quantized<S>, values: &[f32]) -> Result<bool> {
    for _ in 0..100 {
        refine(quantized, values)?;
        if !round_to_nearest_codes(quantized, values) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Round every value to its nearest code on its block's line, and return
/// whether any code moved.
fn round_to_nearest_codes<S: Scale>(quantized: &mut Quantized<S>, values: &[f32]) -> bool {
    let block = quantized.block();
    let mut codes = vec![0i32; quantized.len()];
    unpack_codes(quantized, &mut codes);
    let mut nearest = vec![0i32; quantized.len()];
    let blocks = values.chunks(block).zip(nearest.chunks_mut(block));
    match quantized {
        Quantized::Symmetric {
            scales,
            codes: packed,
            ..
        } => {
            let (zero, bits) = (S::from_f32(0.0), packed.bits());
            for ((block_values, block_codes), &scale) in blocks.zip(scales.iter()) {
                round_block(block_values, block_codes, (scale, zero), bits);
            }
        }
        Quantized::Asymmetric {
            scales,
            zero_points,
            codes: packed,
            ..
        } => {
            let bits = packed.bits();
            let lines = scales.iter().zip(zero_points.iter());
            for ((block_values, block_codes), (&scale, &zero_point)) in blocks.zip(lines) {
                round_block(block_values, block_codes, (scale, zero_point), bits);
            }
        }
        Quantized::Adaptive {
            scales,
            zero_points,
            block_bits,
            ..
        } => {
            let lines = scales.iter().zip(zero_points.iter()).zip(block_bits.iter());
            for ((block_values, block_codes), ((&scale, &zero_point), &bit_width)) in
                blocks.zip(lines)
            {
                round_block(
                    block_values,
                    block_codes,
                    (scale, zero_point),
                    bit_width.into(),
                );
            }
        }
    }
    if nearest == codes {
        return false;
    }
    pack_codes(quantized, &nearest);
    true
}

/// Each value's nearest `bits`-wide code on the line `scale * (code - zero_point)`.
fn round_block<S: Scale>(
    values: &[f32],
    codes: &mut [i32],
    (scale, zero_point): (S, S),
    bits: u32,
) {
    let (scale, zero_point) = (scale.to_f32(), zero_point.to_f32());
    let smallest = smallest_code(bits) as f32;
    let largest = largest_code(bits) as f32;
    for (&value, code) in values.iter().zip(codes) {
        let nearest = (value / scale + zero_point).round();
        *code = nearest.clamp(smallest, largest) as i32;
    }
}

/// Write `codes` back into `quantized`, each block packed at its own width.
fn pack_codes<S: Scale>(quantized: &mut Quantized<S>, codes: &[i32]) {
    match quantized {
        Quantized::Symmetric { codes: packed, .. }
        | Quantized::Asymmetric { codes: packed, .. } => {
            *packed = Packed::from_i32s(codes, packed.bits());
        }
        Quantized::Adaptive {
            codes: bytes,
            block_bits,
            block,
            ..
        } => {
            bytes.clear();
            for (block_codes, &bit_width) in codes.chunks(*block).zip(block_bits.iter()) {
                let packed = Packed::from_i32s(block_codes, bit_width.into());
                bytes.extend_from_slice(packed.as_bytes());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_recovers_known_line() {
        let codes = [0, 1, 2, 3, 4];
        let values: Vec<f32> = codes.iter().map(|&code| 0.5 * code as f32 + 1.0).collect();
        let (scale, zero_point) = fit_scale_and_zero_point(&values, &codes);
        assert!((scale - 0.5).abs() < 1e-5);
        assert!((zero_point + 2.0).abs() < 1e-5);
    }

    #[test]
    fn fit_recovers_line_far_from_zero() {
        let codes: Vec<i32> = (-128..128).collect();
        let values: Vec<f32> = codes
            .iter()
            .map(|&code| 1e-4 * code as f32 + 100.0)
            .collect();
        let (scale, _) = fit_scale_and_zero_point(&values, &codes);
        assert!((scale - 1e-4).abs() < 1e-8, "{scale}");
    }

    #[test]
    fn fit_with_equal_codes_decodes_to_the_mean() {
        let (scale, zero_point) = fit_scale_and_zero_point(&[0.48, 0.5, 0.52], &[7, 7, 7]);
        assert!((scale * (7.0 - zero_point) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn fit_with_equal_values_and_different_codes_stays_near_the_values() {
        assert_eq!(fit_scale_and_zero_point(&[0.0; 3], &[1, 2, 3]), (0.0, 0.0));
        // No line through zero is flat at 0.5, so the fit only comes close.
        let codes = [6, 7, 8];
        let (scale, zero_point) = fit_scale_and_zero_point(&[0.5; 3], &codes);
        assert_eq!(zero_point, 0.0);
        for code in codes {
            let decoded = scale * (code as f32 - zero_point);
            assert!((decoded - 0.5).abs() < 0.1, "{decoded}");
        }
    }

    #[test]
    fn fit_scale_recovers_line_through_zero() {
        let codes = [-8, -3, 0, 5, 7];
        let values: Vec<f32> = codes.iter().map(|&code| 0.25 * code as f32).collect();
        assert!((fit_scale(&values, &codes) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn refine_keeps_a_symmetric_tensor_symmetric() {
        let values: Vec<f32> = (0..256).map(|i| (i as f32).sin()).collect();
        let squared_error = |back: Vec<f32>| -> f32 {
            values
                .iter()
                .zip(back)
                .map(|(a, b)| (a - b) * (a - b))
                .sum()
        };
        let mut q = crate::quantize::<f32, 4, 32>(&values).unwrap();
        let (size, before) = (q.nbytes(), squared_error(q.dequantize()));
        refine(&mut q, &values).unwrap();
        assert!(matches!(q, Quantized::Symmetric { .. }));
        assert_eq!(q.nbytes(), size);
        assert!(squared_error(q.dequantize()) < before);
    }

    #[test]
    fn refine_keeps_flat_block() {
        // Stored as bf16, the fitted zero-point would decode this block as 0.
        let values = [0.3_f32; 32];
        let mut q = crate::asymmetric::quantize::<half::bf16, 8, 32>(&values).unwrap();
        refine(&mut q, &values).unwrap();
        for back in q.dequantize() {
            assert!((back - 0.3).abs() < 1e-3, "{back}");
        }
    }

    #[test]
    fn refine_keeps_the_matrix_shape() {
        let values: Vec<f32> = (0..64).map(|i| i as f32 * 0.01 - 0.3).collect();
        let mut q = crate::quantize::<f32, 8, 32>(&values).unwrap();
        q.set_shape(2, 32).unwrap();
        refine(&mut q, &values).unwrap();
        assert_eq!(q.shape(), Some((2, 32)));
    }

    #[test]
    fn refine_rejects_wrong_length() {
        let mut q = crate::quantize::<f32, 8, 4>(&[0.1, 0.2, 0.3, 0.4]).unwrap();
        assert_eq!(
            refine(&mut q, &[0.1, 0.2]),
            Err(crate::Error::LengthMismatch {
                expected: 4,
                got: 2
            })
        );
    }

    #[test]
    fn alternate_moves_a_value_to_a_closer_code() {
        // Seven small values and one outlier. Once the fit moves the line,
        // 0.02 sits closer to code 6 than to its code 5.
        let values = [0.02_f32, -0.09, 0.10, 0.03, -0.04, 0.13, 0.04, -0.90];
        let mut q = crate::asymmetric::quantize::<f32, 4, 8>(&values).unwrap();
        let mut codes = [0; 8];
        unpack_codes(&q, &mut codes);
        assert_eq!(codes, [5, 4, 7, 6, 5, 7, 6, -8]);
        alternate(&mut q, &values).unwrap();
        unpack_codes(&q, &mut codes);
        assert_eq!(codes, [6, 4, 7, 6, 5, 7, 6, -8]);
    }

    #[test]
    fn alternate_ends_below_refine_and_keeps_each_scheme() {
        // Each block is wider than the last, so adaptive packs them at 4 to 7 bits.
        let values: Vec<f32> = (0..256)
            .map(|i| (i as f32).sin() * (1 + i / 32) as f32)
            .collect();
        let squared_error = |q: &Quantized<half::f16>| -> f32 {
            values
                .iter()
                .zip(q.dequantize())
                .map(|(a, b)| (a - b) * (a - b))
                .sum()
        };
        let tensors: [Quantized<half::f16>; 3] = [
            crate::symmetric::quantize_with(&values, 4, 32).unwrap(),
            crate::asymmetric::quantize_with(&values, 4, 32).unwrap(),
            crate::adaptive::quantize_with(&values, 32, 0.1).unwrap(),
        ];
        for mut original in tensors {
            original.set_shape(8, 32).unwrap();
            let (mut refined, mut alternated) = (original.clone(), original.clone());
            refine(&mut refined, &values).unwrap();
            alternate(&mut alternated, &values).unwrap();
            assert!(squared_error(&alternated) < squared_error(&refined));
            assert_eq!(
                core::mem::discriminant(&alternated),
                core::mem::discriminant(&original)
            );
            assert_eq!(alternated.nbytes(), original.nbytes());
            assert_eq!(alternated.block_bits(), original.block_bits());
            assert_eq!(alternated.shape(), Some((8, 32)));
        }
    }

    #[test]
    fn alternate_stops_once_no_code_moves() {
        let values: Vec<f32> = (0..256).map(|i| (i as f32).sin()).collect();
        let mut q = crate::asymmetric::quantize::<f32, 4, 32>(&values).unwrap();
        assert!(alternate(&mut q, &values).unwrap());
        let settled = q.clone();
        assert!(alternate(&mut q, &values).unwrap());
        assert_eq!(q.codes(), settled.codes());
        assert_eq!(q.scales(), settled.scales());
        assert_eq!(q.zero_points(), settled.zero_points());
    }

    #[test]
    fn alternate_returns_false_until_the_codes_settle() {
        // One scale for 4096 values, most near zero and a few far out. Its
        // codes keep moving for 150 passes, so the first call stops at 100.
        let values: Vec<f32> = (0..4096)
            .map(|i| -((i as f32 + 0.5) / 4096.0).ln())
            .collect();
        let mut q = crate::quantize_tensor::<f32, 6>(&values).unwrap();
        assert!(!alternate(&mut q, &values).unwrap());
        assert!(alternate(&mut q, &values).unwrap());
    }

    #[test]
    fn alternate_rejects_wrong_length() {
        let mut q = crate::quantize::<f32, 8, 4>(&[0.1, 0.2, 0.3, 0.4]).unwrap();
        assert_eq!(
            alternate(&mut q, &[0.1, 0.2]),
            Err(crate::Error::LengthMismatch {
                expected: 4,
                got: 2
            })
        );
    }
}
