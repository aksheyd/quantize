#![allow(clippy::all, unused_mut, dead_code)]
//! Scratch: time one SmolLM-135M decode step's linears at batch 1.
//! Not committed.

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module, Tensor};
use half::f16;
use quantize::{Quantized, quantize};
use std::hint::black_box;
use std::time::Instant;

const HIDDEN: usize = 576;
const KV: usize = 192;
const INTERMEDIATE: usize = 1536;
const VOCAB: usize = 49152;
const LAYERS: usize = 30;

fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut seed = seed;
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed as f32 / u32::MAX as f32) * 2.0 - 1.0) * 0.05
        })
        .collect()
}

struct Layer {
    rows: usize,
    columns: usize,
    ours: Quantized<f16>,
    candle: QMatMul,
}

fn main() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let shapes = [
        (HIDDEN, HIDDEN),
        (KV, HIDDEN),
        (KV, HIDDEN),
        (HIDDEN, HIDDEN),
        (INTERMEDIATE, HIDDEN),
        (INTERMEDIATE, HIDDEN),
        (HIDDEN, INTERMEDIATE),
    ];
    let mut layers = Vec::new();
    // Distinct weights per layer, so the whole model streams from memory as in a real run.
    let only = std::env::var("ONLY").ok();
    let layer_count = if only.is_some() { 1 } else { LAYERS };
    for layer in 0..layer_count {
        for (index, &(rows, columns)) in shapes.iter().enumerate() {
            let weights = values(rows * columns, (layer * 7 + index) as u32 + 1);
            let mut ours = quantize::<f16, 4, 32>(&weights).unwrap();
            ours.set_shape(rows, columns).unwrap();
            let tensor = Tensor::from_slice(&weights, (rows, columns), &device)?;
            let candle = QMatMul::from_qtensor(QTensor::quantize(&tensor, GgmlDType::Q4_0)?)?;
            layers.push(Layer {
                rows,
                columns,
                ours,
                candle,
            });
        }
    }
    let lm_weights = values(VOCAB * HIDDEN, 999);
    let mut lm_ours = quantize::<f16, 4, 32>(&lm_weights).unwrap();
    lm_ours.set_shape(VOCAB, HIDDEN).unwrap();
    let lm_tensor = Tensor::from_slice(&lm_weights, (VOCAB, HIDDEN), &device)?;
    let lm_candle = QMatMul::from_qtensor(QTensor::quantize(&lm_tensor, GgmlDType::Q4_0)?)?;
    drop(lm_weights);
    layers.push(Layer {
        rows: VOCAB,
        columns: HIDDEN,
        ours: lm_ours,
        candle: lm_candle,
    });

    let inputs: Vec<Vec<f32>> = [HIDDEN, INTERMEDIATE]
        .iter()
        .map(|&n| values(n, 4242))
        .collect();
    let input_for = |columns: usize| -> &[f32] {
        if columns == HIDDEN {
            &inputs[0]
        } else {
            &inputs[1]
        }
    };
    let candle_inputs: Vec<Tensor> = [HIDDEN, INTERMEDIATE]
        .iter()
        .map(|&n| Tensor::from_slice(&values(n, 4242), (1, n), &device).unwrap())
        .collect();
    let candle_input_for = |columns: usize| -> &Tensor {
        if columns == HIDDEN {
            &candle_inputs[0]
        } else {
            &candle_inputs[1]
        }
    };

    let which = std::env::var("WHICH").unwrap_or_else(|_| "both".into());
    let repeats: usize = std::env::var("REPEATS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);

    let mut time_ours = || {
        let start = Instant::now();
        for layer in &layers {
            let out = layer.ours.matmul(input_for(layer.columns)).unwrap();
            debug_assert_eq!(out.len(), layer.rows);
            black_box(out);
        }
        start.elapsed().as_secs_f64()
    };
    let mut time_candle = || {
        let start = Instant::now();
        for layer in &layers {
            let out = layer
                .candle
                .forward(candle_input_for(layer.columns))
                .unwrap();
            black_box(out);
        }
        start.elapsed().as_secs_f64()
    };

    let median = |mut samples: Vec<f64>| {
        samples.sort_by(f64::total_cmp);
        samples[samples.len() / 2]
    };
    let report = |name: &str, seconds: f64| {
        println!(
            "{name:<24} {:>8.2} ms/token  {:>7.1} tok/s (linears only)",
            seconds * 1e3,
            1.0 / seconds
        );
    };
    if which == "both" || which == "ours" {
        for _ in 0..3 {
            time_ours();
        }
        let samples: Vec<f64> = (0..repeats).map(|_| time_ours()).collect();
        report("quantize Q4_32 f16", median(samples));
    }
    if which == "both" || which == "candle" {
        for _ in 0..3 {
            time_candle();
        }
        let samples: Vec<f64> = (0..repeats).map(|_| time_candle()).collect();
        report("candle Q4_0", median(samples));
    }

    // Per-shape breakdown for ours.
    if which == "both" || which == "ours" {
        let mut per_shape = std::collections::BTreeMap::new();
        for layer in &layers {
            let input = input_for(layer.columns);
            let mut samples = Vec::new();
            for _ in 0..21 {
                let start = Instant::now();
                black_box(layer.ours.matmul(input).unwrap());
                samples.push(start.elapsed().as_secs_f64());
            }
            let entry = per_shape
                .entry((layer.rows, layer.columns))
                .or_insert((0.0_f64, 0usize));
            entry.0 += median(samples);
            entry.1 += 1;
        }
        for ((rows, columns), (seconds, count)) in per_shape {
            let values = (rows * columns * count) as f64;
            println!(
                "  ours {rows:>6}x{columns:<5} x{count:<3} {:>7.3} ms  {:.3} ns/value",
                seconds * 1e3,
                seconds * 1e9 / values
            );
        }
    }
    Ok(())
}
