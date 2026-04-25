//! Per-tensor scale and round-trip helpers used by `Harness::run`.

use super::{DequantizeFn, QuantizeFn};

/// Pick a single per-tensor scale: largest |x| maps to ~127.
fn pick_scale(values: &[f32]) -> f32 {
    let max_abs = values.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    if max_abs > 0.0 {
        max_abs / 127.0
    } else {
        1.0
    }
}

pub(super) fn roundtrip(
    values: &[f32],
    quantize: QuantizeFn,
    dequantize: DequantizeFn,
) -> Vec<f32> {
    let scale = pick_scale(values);
    values
        .iter()
        .map(|&x| dequantize(quantize(x, scale), scale))
        .collect()
}
