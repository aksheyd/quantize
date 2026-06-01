//! Test harness. `Harness::new(matrix_size, runs).run()` generates `runs`
//! fresh random matrices, evaluates every method, returns mean ± std stats.

mod methods;
mod metrics;
mod new;
mod quant;
mod run;
mod sample;
mod stats;

use candle_core::{Device, Tensor};

#[allow(dead_code)]
pub struct Stats {
    pub mse_mean: f32,
    pub mse_std: f32,
    pub cosine_mean: f32,
    pub cosine_std: f32,
}

pub struct MethodReport {
    pub name: &'static str,
    pub bits_per_element: f32,
    pub stats: Stats,
}

pub struct Comparison {
    pub matrix_size: usize,
    pub runs: usize,
    pub methods: Vec<MethodReport>,
}

pub struct Harness {
    matrix_size: usize,
    runs: usize,
    device: Device,
}

// One run's worth of fresh data. Submodules can see the private fields
// because Rust lets descendant modules access the parent's private items.
struct Sample {
    matrix_size: usize,
    matrix_a: Vec<f32>,
    matrix_b: Vec<f32>,
    tensor_a: Tensor,
    tensor_b: Tensor,
    ground_truth: Vec<f32>,
}
