//! Test harness. `Harness::new(matrix_size, runs).run()` generates `runs`
//! fresh random matrices, evaluates every method, returns mean MSE.

mod methods;
mod metrics;
mod new;
mod quant;
mod run;
mod sample;

use candle_core::{Device, Tensor};

pub struct MethodReport {
    pub bits_per_element: f32,
    pub mse: f32,
}

pub struct Comparison {
    pub methods: Vec<MethodReport>,
}

pub struct Harness {
    matrix_size: usize,
    runs: usize,
    device: Device,
}

struct Sample {
    matrix_size: usize,
    matrix_a: Vec<f32>,
    matrix_b: Vec<f32>,
    tensor_a: Tensor,
    tensor_b: Tensor,
    ground_truth: Vec<f32>,
}
