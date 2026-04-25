//! Per-tensor round-trip helpers used by `Harness::run`.

use quantize::{choose_scale_bits, dequantize_bits, quantize_bits};

pub(super) fn roundtrip(values: &[f32]) -> Vec<f32> {
    let scale = choose_scale_bits::<8>(values);
    values
        .iter()
        .map(|&x| dequantize_bits(quantize_bits::<8>(x, scale), scale))
        .collect()
}
