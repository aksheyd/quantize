//! After codes are chosen, pick a better scale and zero-point.
//!
//! We want: `original ≈ scale * (code - zero_point)`.
//! Same as: `original ≈ scale * code + offset`, with
//! `offset = -scale * zero_point`. Codes stay fixed.

use crate::decode::unpack_codes;
use crate::scale::Scale;
use crate::tensor::Quantized;

/// Best-fit `scale` and `zero_point` for `values ≈ scale * (codes - zero_point)`.
///
/// Returns `(1.0, 0.0)` when every code is the same (nothing to fit).
pub fn fit_scale_and_zero_point(values: &[f32], codes: &[i32]) -> (f32, f32) {
    debug_assert_eq!(values.len(), codes.len());
    let count = values.len() as f32;
    if count == 0.0 {
        return (1.0, 0.0);
    }

    let mut sum_codes = 0.0_f32;
    let mut sum_values = 0.0_f32;
    let mut sum_code_squared = 0.0_f32;
    let mut sum_code_times_value = 0.0_f32;
    for (&value, &code) in values.iter().zip(codes) {
        let code = code as f32;
        sum_codes += code;
        sum_values += value;
        sum_code_squared += code * code;
        sum_code_times_value += code * value;
    }

    let mean_code = sum_codes / count;
    let mean_value = sum_values / count;
    let code_spread = sum_code_squared - sum_codes * mean_code;
    if code_spread.abs() < 1e-12 {
        return (1.0, 0.0);
    }

    // Line of best fit: value ≈ scale * code + offset.
    let scale = (sum_code_times_value - sum_codes * mean_value) / code_spread;
    let offset = mean_value - scale * mean_code;
    if scale.abs() < 1e-12 {
        return (1.0, 0.0);
    }
    let zero_point = -offset / scale;
    (scale, zero_point)
}

/// Recompute each block's scale and zero-point. Codes do not change.
///
/// Empty input is left as-is. [`Quantized::Symmetric`] becomes
/// [`Quantized::Asymmetric`].
pub fn refine<S: Scale>(quantized: &mut Quantized<S>, values: &[f32]) {
    if quantized.is_empty() || values.len() != quantized.len() {
        return;
    }
    let block = quantized.block();
    let mut codes = vec![0i32; quantized.len()];
    unpack_codes(quantized, &mut codes);

    let mut scales = Vec::new();
    let mut zero_points = Vec::new();
    for (block_index, block_values) in values.chunks(block).enumerate() {
        let start = block_index * block;
        let block_codes = &codes[start..start + block_values.len()];
        let (scale, zero_point) = fit_scale_and_zero_point(block_values, block_codes);
        scales.push(S::from_f32(scale));
        zero_points.push(S::from_f32(zero_point));
    }

    *quantized = match quantized {
        Quantized::Adaptive {
            bytes, bits, len, ..
        } => Quantized::Adaptive {
            scales,
            zero_points,
            bytes: bytes.clone(),
            bits: bits.clone(),
            block,
            len: *len,
        },
        Quantized::Symmetric { codes, len, .. } | Quantized::Asymmetric { codes, len, .. } => {
            Quantized::Asymmetric {
                scales,
                zero_points,
                codes: codes.clone(),
                block,
                len: *len,
            }
        }
    };
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
}
