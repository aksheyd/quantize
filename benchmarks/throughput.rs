//! Quantize / dequantize / fused-dot throughput vs candle.
//!
//! Run: `cargo run --release --example throughput`

use candle_core::{
    quantized::{GgmlDType, QTensor},
    Device, Tensor,
};
use half::f16;
use quantize::quantize;
use std::hint::black_box;
use std::time::Instant;

const SIDE: usize = 1024;
const N: usize = SIDE * SIDE;
const ITERS: usize = 50;

fn main() -> candle_core::Result<()> {
    let mut seed = 0x1234_5678u32;
    let values: Vec<f32> = (0..N)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();

    println!("n = {N} ({SIDE}x{SIDE}), iters = {ITERS}\n");
    println!("{:<22}{:>12}{:>14}", "kernel", "ns/elt", "GB/s (in)");
    println!("{:-<22}{:->12}{:->14}", "", "", "");

    bench("quantize 4b×32", || {
        black_box(quantize::<f16, 4, 32>(&values).unwrap());
    });
    bench("quantize 8b×32", || {
        black_box(quantize::<f16, 8, 32>(&values).unwrap());
    });

    let q4 = quantize::<f16, 4, 32>(&values).unwrap();
    let q8 = quantize::<f16, 8, 32>(&values).unwrap();
    let mut out = vec![0.0f32; N];
    bench("dequant 4b×32", || {
        q4.dequantize_into(&mut out).unwrap();
        black_box(&out);
    });
    bench("dequant 8b×32", || {
        q8.dequantize_into(&mut out).unwrap();
        black_box(&out);
    });
    bench("dot 8b×32", || {
        black_box(q8.dot(&values).unwrap());
    });

    let device = Device::Cpu;
    let t = Tensor::from_vec(values.clone(), N, &device)?;
    bench("candle Q4_0 quant", || {
        black_box(QTensor::quantize(&t, GgmlDType::Q4_0).unwrap());
    });
    bench("candle Q8_0 quant", || {
        black_box(QTensor::quantize(&t, GgmlDType::Q8_0).unwrap());
    });
    let cq4 = QTensor::quantize(&t, GgmlDType::Q4_0)?;
    let cq8 = QTensor::quantize(&t, GgmlDType::Q8_0)?;
    bench("candle Q4_0 dequant", || {
        black_box(cq4.dequantize(&device).unwrap());
    });
    bench("candle Q8_0 dequant", || {
        black_box(cq8.dequantize(&device).unwrap());
    });
    Ok(())
}

fn bench(name: &str, mut f: impl FnMut()) {
    for _ in 0..4 {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
        f();
    }
    let ns = t0.elapsed().as_secs_f64() * 1e9 / (ITERS as f64 * N as f64);
    let gbs = (N as f64 * 4.0) / (ns * N as f64); // bytes in / time, in GB/s
    println!("{name:<22}{ns:>12.3}{gbs:>14.2}");
}
