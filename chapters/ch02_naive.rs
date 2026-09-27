//! # Chapter 2 — naive (max-abs symmetric, 8-bit)
//!
//! **Previously** (`ch01_simple`): we just cast f32 to i8.
//!
//! **Problem**: an `i8` holds only whole numbers, so the cast drops the
//! fraction. ML weights live in roughly `[-1, 1]`, so `0.42 as i8` became `0`.
//! Every weight vanished.
//!
//! **Fix**: divide by a *scale* first, so the largest magnitude in the tensor
//! maps to ~127, then round. The integer we store is called a *code*. Now the
//! codes use the full i8 range, and every weight comes back close to where it
//! started.
//!
//! **Still wrong**: one giant outlier wrecks the scale for everyone else,
//! and we're stuck at 8 bits.
//!
//! Run it: `cargo run --release --example ch02_naive`

fn quantize_naive(x: f32, scale: f32) -> i8 {
    (x / scale).round().clamp(-128.0, 127.0) as i8
}

fn dequantize_naive(code: i8, scale: f32) -> f32 {
    code as f32 * scale
}

fn choose_scale_naive(values: &[f32]) -> f32 {
    let max_abs = values.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    if max_abs > 0.0 {
        max_abs / 127.0
    } else {
        1.0
    }
}

/// The biggest gap between an input and what came back.
fn worst_error(inputs: &[f32], outputs: &[f32]) -> f32 {
    let mut worst = 0.0_f32;
    for (input, output) in inputs.iter().zip(outputs) {
        worst = worst.max((input - output).abs());
    }
    worst
}

fn main() {
    let weights = [0.42_f32, -0.10, 0.70, -0.50, 0.99, -0.99];
    let scale = choose_scale_naive(&weights);

    println!("scale = {scale:.6}\n");
    println!("{:>8}  {:>4}  {:>10}", "input", "code", "back→f32");
    println!("{:>8}  {:>4}  {:>10}", "-----", "----", "--------");
    let mut reconstructed = Vec::new();
    for &w in &weights {
        let code = quantize_naive(w, scale);
        let back = dequantize_naive(code, scale);
        println!("{w:>8.2}  {code:>4}  {back:>10.4}");
        reconstructed.push(back);
    }
    println!(
        "\nworst error: {:.4}",
        worst_error(&weights, &reconstructed)
    );

    println!("\nNo zero-collapse this time. Chapter 3 (`ch03_bits`) generalizes");
    println!("this same algorithm to any bit width (4-bit, 16-bit, etc.).");
}
