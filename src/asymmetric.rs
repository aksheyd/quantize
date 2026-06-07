//! Asymmetric quantization.
//!
//! A scale and a zero-point per group. The integer codes are shifted
//! so the full range maps exactly to the group's actual min and max.
//! This avoids wasting codes on the unused side of zero.
//!
//! A "group" can be a fixed-size block or the entire tensor.
//!
//! When using blocks, you can supply a different bit width per block.
//! This is the "precision-aware" part: blocks with small ranges can use
//! fewer bits while still meeting a target error tolerance.

use crate::Scale;

fn max_int(bits: u32) -> i32 {
    (1_i32 << (bits - 1)) - 1
}
fn min_int(bits: u32) -> i32 {
    -(1_i32 << (bits - 1))
}

/// Quantize into fixed-size blocks, using the given bit width for each block.
/// `bits` must have one entry per block (or fewer; missing blocks default to 8).
/// Returns `(scales, zero_points, codes)`.
pub fn quantize<S: Scale, const BLOCK: usize>(
    values: &[f32],
    bits: &[u32],
) -> (Vec<S>, Vec<S>, Vec<i32>) {
    let mut scales = Vec::with_capacity(values.len() / BLOCK + 1);
    let mut zps = Vec::with_capacity(values.len() / BLOCK + 1);
    let mut codes = Vec::with_capacity(values.len());
    for (i, chunk) in values.chunks(BLOCK).enumerate() {
        let b = *bits.get(i).unwrap_or(&8);
        let rmin = chunk.iter().copied().fold(f32::INFINITY, f32::min);
        let rmax = chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let (sf, zpf) = if rmin >= rmax {
            (1.0, 0.0)
        } else {
            let qmin = min_int(b) as f32;
            let qmax = max_int(b) as f32;
            let scale = (rmax - rmin) / (qmax - qmin);
            let zp = qmin - rmin / scale;
            (scale, zp)
        };
        scales.push(S::from_f32(sf));
        zps.push(S::from_f32(zpf));
        let qmin = min_int(b);
        let qmax = max_int(b);
        codes.extend(
            chunk
                .iter()
                .map(|&x| ((x / sf + zpf).round() as i32).clamp(qmin, qmax)),
        );
    }
    (scales, zps, codes)
}

/// Reconstruct from per-block scales and zero-points.
pub fn dequantize<S: Scale, const BLOCK: usize>(
    scales: &[S],
    zero_points: &[S],
    codes: &[i32],
) -> Vec<f32> {
    codes
        .chunks(BLOCK)
        .zip(scales.iter().zip(zero_points))
        .flat_map(|(blk, (&s, &zp))| {
            let sf = s.to_f32();
            let zpf = zp.to_f32();
            blk.iter().map(move |&q| (q as f32 - zpf) * sf)
        })
        .collect()
}

/// Quantize the entire tensor with one scale and one zero-point.
/// `bits` sets the integer width for the whole tensor.
/// Returns `(scale, zero_point, codes)`.
pub fn quantize_tensor<S: Scale>(values: &[f32], bits: u32) -> (S, S, Vec<i32>) {
    if values.is_empty() {
        let z = S::from_f32(0.0);
        return (S::from_f32(1.0), z, vec![]);
    }
    let rmin = values.iter().copied().fold(f32::INFINITY, f32::min);
    let rmax = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let (sf, zpf) = if rmin >= rmax {
        (1.0, 0.0)
    } else {
        let qmin = min_int(bits) as f32;
        let qmax = max_int(bits) as f32;
        let scale = (rmax - rmin) / (qmax - qmin);
        let zp = qmin - rmin / scale;
        (scale, zp)
    };
    let s = S::from_f32(sf);
    let zp = S::from_f32(zpf);
    let qmin = min_int(bits);
    let qmax = max_int(bits);
    let codes = values
        .iter()
        .map(|&x| ((x / sf + zpf).round() as i32).clamp(qmin, qmax))
        .collect();
    (s, zp, codes)
}

/// Reconstruct the entire tensor from a single scale and zero-point.
pub fn dequantize_tensor<S: Scale>(scale: S, zero_point: S, codes: &[i32]) -> Vec<f32> {
    let sf = scale.to_f32();
    let zpf = zero_point.to_f32();
    codes.iter().map(|&q| (q as f32 - zpf) * sf).collect()
}
