//! # Chapter 4 — block (per-block scales)
//!
//! **Previously** (`ch03_bits`): we could quantize to any bit width, but we still
//! used a *single* scale for the entire tensor.
//! **Problem**: One outlier anywhere in a long vector forces a huge scale for
//! everyone. All the normal small values lose precision.
//! **Fix**: Split the data into small fixed-size blocks and pick a separate
//! scale for each block. An outlier now only pollutes its own small neighborhood.
//! **This is the algorithm** now living in `src/block.rs` (with extra polish for
//! `f16`/`bf16` scales and const generics).
//!
//! Run it: `cargo run --release --example ch04_block`

const fn max_int<const BITS: u32>() -> i32 {
    (1_i32 << (BITS - 1)) - 1
}

const fn min_int<const BITS: u32>() -> i32 {
    -(1_i32 << (BITS - 1))
}

fn choose_scale<const BITS: u32>(values: &[f32]) -> f32 {
    let max_abs = values.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    if max_abs > 0.0 {
        max_abs / max_int::<BITS>() as f32
    } else {
        1.0
    }
}

fn quantize_val<const BITS: u32>(x: f32, scale: f32) -> i32 {
    let lo = min_int::<BITS>() as f32;
    let hi = max_int::<BITS>() as f32;
    (x / scale).round().clamp(lo, hi) as i32
}

fn dequantize_val(q: i32, scale: f32) -> f32 {
    q as f32 * scale
}

/// Quantize with one scale for the whole slice (what we had in Chapter 3).
fn quantize_tensor<const BITS: u32>(values: &[f32]) -> (f32, Vec<i32>) {
    let scale = choose_scale::<BITS>(values);
    let codes = values
        .iter()
        .map(|&x| quantize_val::<BITS>(x, scale))
        .collect();
    (scale, codes)
}

/// Quantize with one scale per BLOCK-sized chunk.
fn quantize_blocks<const BITS: u32, const BLOCK: usize>(values: &[f32]) -> (Vec<f32>, Vec<i32>) {
    let mut scales = Vec::new();
    let mut codes = Vec::new();

    for block in values.chunks(BLOCK) {
        let scale = choose_scale::<BITS>(block);
        scales.push(scale);
        for &x in block {
            codes.push(quantize_val::<BITS>(x, scale));
        }
    }
    (scales, codes)
}

fn dequantize_blocks(codes: &[i32], scales: &[f32], block_size: usize) -> Vec<f32> {
    codes
        .chunks(block_size)
        .zip(scales)
        .flat_map(|(blk, &s)| blk.iter().map(move |&q| dequantize_val(q, s)))
        .collect()
}

fn main() {
    // A small tensor with a clear outlier in the middle block.
    // Most values are tiny (~0.05). One block contains a huge 4.0 spike.
    let weights = [
        0.04_f32, 0.05, -0.03, 0.06, // block 0 — tiny
        0.02, 0.07, 0.01, -0.04, // block 1 — tiny
        4.0, 0.10, -0.05, 0.08, // block 2 — contains huge outlier
        0.03, -0.02, 0.06, 0.04, // block 3 — tiny again
    ];

    const BITS: u32 = 4;
    const BLOCK: usize = 4;

    println!("Input (16 values, one obvious outlier in block 2):\n");

    // --- Per-tensor (Ch. 3 style) ---
    let (tensor_scale, tensor_codes) = quantize_tensor::<BITS>(&weights);
    let tensor_back: Vec<f32> = tensor_codes
        .iter()
        .map(|&q| dequantize_val(q, tensor_scale))
        .collect();

    println!(
        "Per-tensor (1 scale for everything) — scale = {:.4}",
        tensor_scale
    );
    println!("{:>5}  {:>6}  {:>8}  {:>8}", "idx", "input", "code", "back");
    println!("{:-<5}  {:-<6}  {:-<8}  {:-<8}", "", "", "", "");
    for (i, ((&w, &q), &b)) in weights
        .iter()
        .zip(&tensor_codes)
        .zip(&tensor_back)
        .enumerate()
    {
        println!("{:>5}  {:>6.2}  {:>8}  {:>8.3}", i, w, q, b);
    }

    let tensor_mse: f32 = weights
        .iter()
        .zip(&tensor_back)
        .map(|(w, b)| (w - b).powi(2))
        .sum::<f32>()
        / weights.len() as f32;
    println!("MSE vs original: {:.6}\n", tensor_mse);

    // --- Per-block (this chapter) ---
    let (block_scales, block_codes) = quantize_blocks::<BITS, BLOCK>(&weights);
    let block_back = dequantize_blocks(&block_codes, &block_scales, BLOCK);

    println!("Per-block ({} scales, BLOCK={})", block_scales.len(), BLOCK);
    println!(
        "scales: {:?}",
        block_scales
            .iter()
            .map(|s| format!("{:.4}", s))
            .collect::<Vec<_>>()
    );
    println!("{:>5}  {:>6}  {:>8}  {:>8}", "idx", "input", "code", "back");
    println!("{:-<5}  {:-<6}  {:-<8}  {:-<8}", "", "", "", "");
    for (i, ((&w, &q), &b)) in weights
        .iter()
        .zip(&block_codes)
        .zip(&block_back)
        .enumerate()
    {
        println!("{:>5}  {:>6.2}  {:>8}  {:>8.3}", i, w, q, b);
    }

    let block_mse: f32 = weights
        .iter()
        .zip(&block_back)
        .map(|(w, b)| (w - b).powi(2))
        .sum::<f32>()
        / weights.len() as f32;
    println!("MSE vs original: {:.6}\n", block_mse);

    println!("Notice: the outlier only destroyed precision inside its own block.");
    println!("All the other tiny values kept much better fidelity.");

    println!("\nThe code above is the heart of `src/block.rs`.");
    println!("The real library adds const generics for BITS and BLOCK, plus the");
    println!("`Scale` trait so you can store scales as f16/bf16 instead of f32.");
}
