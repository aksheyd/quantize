/// Map a f32 to a i8.
pub fn quantize(x: f32, _scale: f32) -> i8 {
    x as i8
}

/// Map a i8 to a f32.
pub fn dequantize(x: i8, _scale: f32) -> f32 {
    x as f32
}
