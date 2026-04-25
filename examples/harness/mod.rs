//! Test harness for quantization experiments.
//!
//! `Harness::new(matrix_size)` builds the random matrices and precomputes the
//! two reference results (f32 ground truth and Candle Q8_0). Call
//! `run(quantize, dequantize)` to evaluate any pair of quant/dequant
//! functions against them.

mod metrics;
mod new;
mod quant;
mod run;

use candle_core::Device;

pub type QuantizeFn = fn(f32, f32) -> i8;
pub type DequantizeFn = fn(i8, f32) -> f32;

pub struct Stats {
    pub mse: f32,
    pub cosine: f32,
}

pub struct Comparison {
    pub matrix_size: usize,
    pub quantize: Stats,
    pub candle_q8_0: Stats,
}

pub struct Harness {
    // input
    matrix_size: usize,

    // computed
    device: Device, // CPU or GPU
    matrix_a: Vec<f32>,
    matrix_b: Vec<f32>,
    correct_matmul: Vec<f32>,
    candle_matmul: Vec<f32>,
}
