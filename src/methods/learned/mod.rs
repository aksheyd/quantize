//! After codes are chosen, pick a better scale and zero-point.
//!
//! We want: `original ≈ scale * (code - zero_point)`.
//! Same as: `original ≈ scale * code + offset`, with
//! `offset = -scale * zero_point`. Codes stay fixed.
//!
//! A symmetric block has no zero-point, so its line must pass through zero:
//! `original ≈ scale * code`, and only the scale is fitted.

use crate::decode::unpack_codes;
use crate::error::{check_len, Result};
use crate::scale::Scale;
use crate::tensor::Quantized;

/// Best-fit `scale` and `zero_point` for `values ≈ scale * (codes - zero_point)`.
///
/// When every code is the same there is no slope to fit, so the scale stays
/// `1.0` and only the offset is fitted: every code decodes to the mean value.
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
        return (1.0, 0.0);
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
/// the new ones decode it better. Empty input is left as-is.
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
        // Stored as bf16, the fitted zero-point would decode this block as 0.5.
        let values = [0.3_f32; 32];
        let mut q = crate::asymmetric::quantize::<half::bf16, 8, 32>(&values).unwrap();
        refine(&mut q, &values).unwrap();
        for back in q.dequantize() {
            assert!((back - 0.3).abs() < 1e-3, "{back}");
        }
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
}
