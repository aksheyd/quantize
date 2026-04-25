//! Symmetric quantization, any bit width.
//!
//! Pick `BITS` (e.g. 4, 8, 16) and the integer range becomes
//! `-(2^(BITS-1)) ..= (2^(BITS-1)) - 1`. Same algorithm as `ch02_naive`,
//! just generalized.

/// Largest signed integer representable in `BITS` bits, e.g. 127 for 8.
const fn max_int<const BITS: u32>() -> i32 {
    (1_i32 << (BITS - 1)) - 1
}

/// Smallest signed integer representable in `BITS` bits, e.g. -128 for 8.
const fn min_int<const BITS: u32>() -> i32 {
    -(1_i32 << (BITS - 1))
}

/// Map an f32 to a `BITS`-bit signed integer using an explicit scale.
pub fn quantize_bits<const BITS: u32>(x: f32, scale: f32) -> i32 {
    (x / scale)
        .round()
        .clamp(min_int::<BITS>() as f32, max_int::<BITS>() as f32) as i32
}

/// Map a quantized integer back to f32. Bit width doesn't matter here.
pub fn dequantize_bits(q: i32, scale: f32) -> f32 {
    q as f32 * scale
}

/// Pick a per-tensor scale: largest |x| maps to the top of the int range.
pub fn choose_scale_bits<const BITS: u32>(values: &[f32]) -> f32 {
    let max_abs = values.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    if max_abs > 0.0 {
        max_abs / max_int::<BITS>() as f32
    } else {
        1.0
    }
}
