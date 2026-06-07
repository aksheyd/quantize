//! Symmetric quantization.
//!
//! One scale per group, chosen from the max absolute value.
//! Integer codes are always symmetric around zero.
//!
//! A "group" can be either a fixed-size block or the entire tensor.
//! Use the block form when you want to limit the impact of outliers within the tensor.
//! Use the tensor form for the simplest possible case (one scale total).
//!
//! This is the baseline before introducing zero-points (asymmetric).

use crate::Scale;

const fn max_int<const BITS: u32>() -> i32 {
    (1_i32 << (BITS - 1)) - 1
}
const fn min_int<const BITS: u32>() -> i32 {
    -(1_i32 << (BITS - 1))
}

fn choose_scale<const BITS: u32>(group: &[f32]) -> f32 {
    let max_abs = group.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    if max_abs > 0.0 {
        max_abs / max_int::<BITS>() as f32
    } else {
        1.0
    }
}

/// Quantize into fixed-size blocks. One scale per block.
/// Returns `(scales, codes)`.
pub fn quantize<S: Scale, const BITS: u32, const BLOCK: usize>(
    values: &[f32],
) -> (Vec<S>, Vec<i32>) {
    let mut scales = Vec::with_capacity(values.len() / BLOCK + 1);
    let mut codes = Vec::with_capacity(values.len());
    for chunk in values.chunks(BLOCK) {
        let s = S::from_f32(choose_scale::<BITS>(chunk));
        scales.push(s);
        let sf = s.to_f32();
        let lo = min_int::<BITS>() as f32;
        let hi = max_int::<BITS>() as f32;
        codes.extend(chunk.iter().map(|&x| (x / sf).round().clamp(lo, hi) as i32));
    }
    (scales, codes)
}

/// Reconstruct from per-block scales and codes.
pub fn dequantize<S: Scale, const BLOCK: usize>(scales: &[S], codes: &[i32]) -> Vec<f32> {
    codes
        .chunks(BLOCK)
        .zip(scales)
        .flat_map(|(blk, s)| {
            let sf = s.to_f32();
            blk.iter().map(move |&q| q as f32 * sf)
        })
        .collect()
}

/// Quantize the entire tensor with a single scale.
/// Returns `(scale, codes)`.
pub fn quantize_tensor<S: Scale, const BITS: u32>(values: &[f32]) -> (S, Vec<i32>) {
    if values.is_empty() {
        return (S::from_f32(1.0), vec![]);
    }
    let s = S::from_f32(choose_scale::<BITS>(values));
    let sf = s.to_f32();
    let lo = min_int::<BITS>() as f32;
    let hi = max_int::<BITS>() as f32;
    let codes = values
        .iter()
        .map(|&x| (x / sf).round().clamp(lo, hi) as i32)
        .collect();
    (s, codes)
}

/// Reconstruct the entire tensor from a single scale and codes.
pub fn dequantize_tensor<S: Scale>(scale: S, codes: &[i32]) -> Vec<f32> {
    let sf = scale.to_f32();
    codes.iter().map(|&q| q as f32 * sf).collect()
}
