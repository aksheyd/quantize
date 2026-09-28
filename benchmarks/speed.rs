//! Times quantize and dequantize in this crate and in candle the same way:
//! on the same values, with the same timer, and with every call allocating
//! its output. `throughput.rs` prints these times, and `update_readme.rs`
//! writes them into README.md's speed table.

use candle_core::{
    quantized::{GgmlDType, QTensor},
    Device, Result, Tensor,
};
use half::f16;
use quantize::quantize;
use std::hint::black_box;
use std::time::Instant;

pub const SIDE: usize = 1024;
pub const ITERATIONS: usize = 50;

/// One kernel's time in each library, in nanoseconds per value.
#[derive(Clone, Copy)]
pub struct KernelTime {
    pub name: &'static str,
    pub this_crate: f64,
    pub candle: f64,
}

/// `SIDE * SIDE` values in -1..1 from a fixed seed, so every run times the same data.
pub fn values() -> Vec<f32> {
    let mut seed = 0x1234_5678u32;
    (0..SIDE * SIDE)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Candle's `QTensor` can only dequantize into a new tensor, so this crate's
/// side calls `dequantize()`, which allocates too, not `dequantize_into`.
pub fn measure(values: &[f32]) -> Result<[KernelTime; 4]> {
    let device = Device::Cpu;
    let tensor = Tensor::from_slice(values, values.len(), &device)?;
    let ours_4bit = quantize::<f16, 4, 32>(values).unwrap();
    let ours_8bit = quantize::<f16, 8, 32>(values).unwrap();
    let candle_4bit = QTensor::quantize(&tensor, GgmlDType::Q4_0)?;
    let candle_8bit = QTensor::quantize(&tensor, GgmlDType::Q8_0)?;

    Ok([
        KernelTime {
            name: "4-bit quant",
            this_crate: time_per_value(|| quantize::<f16, 4, 32>(values).unwrap()),
            candle: time_per_value(|| QTensor::quantize(&tensor, GgmlDType::Q4_0).unwrap()),
        },
        KernelTime {
            name: "8-bit quant",
            this_crate: time_per_value(|| quantize::<f16, 8, 32>(values).unwrap()),
            candle: time_per_value(|| QTensor::quantize(&tensor, GgmlDType::Q8_0).unwrap()),
        },
        KernelTime {
            name: "4-bit dequant",
            this_crate: time_per_value(|| ours_4bit.dequantize()),
            candle: time_per_value(|| candle_4bit.dequantize(&device).unwrap()),
        },
        KernelTime {
            name: "8-bit dequant",
            this_crate: time_per_value(|| ours_8bit.dequantize()),
            candle: time_per_value(|| candle_8bit.dequantize(&device).unwrap()),
        },
    ])
}

/// Nanoseconds per value for one call of `f`: the median of `ITERATIONS`
/// calls after 4 warm-up calls, so a call stalled by another process on the
/// machine doesn't skew the result.
pub fn time_per_value<T>(mut f: impl FnMut() -> T) -> f64 {
    for _ in 0..4 {
        black_box(f());
    }
    let mut seconds: Vec<f64> = (0..ITERATIONS)
        .map(|_| {
            let start = Instant::now();
            black_box(f());
            start.elapsed().as_secs_f64()
        })
        .collect();
    seconds.sort_by(f64::total_cmp);
    seconds[ITERATIONS / 2] * 1e9 / (SIDE * SIDE) as f64
}
