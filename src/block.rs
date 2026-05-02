//! Block quantization, any bit width and scale precision.
//!
//! Split values into fixed-size blocks, pick a scale per block.
//! `BITS` sets integer precision, `BLOCK` sets the block size,
//! and the `Scale` type sets scale storage precision (f32, f16, …).

use crate::Scale;

const fn max_int<const BITS: u32>() -> i32 {
    (1_i32 << (BITS - 1)) - 1
}
const fn min_int<const BITS: u32>() -> i32 {
    -(1_i32 << (BITS - 1))
}

fn choose_scale<const BITS: u32>(block: &[f32]) -> f32 {
    let max_abs = block.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    if max_abs > 0.0 {
        max_abs / max_int::<BITS>() as f32
    } else {
        1.0
    }
}

/// Quantize a slice into blocks. Returns `(scales, codes)`.
/// `scales` has one `S` entry per block; `codes` has one i32 per input element.
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

/// Reconstruct f32 values from block-quantized codes and scales.
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
