//! # Chapter 1 — simple (the strawman)
//!
//! This is the dumbest possible "quantization": just cast the f32 to an i8.
//! No scale, no rounding logic, nothing clever.
//!
//! Run it: `cargo run --release --example ch01_simple`
//!
//! Watch what happens to a handful of typical ML weights (values in `[-1, 1]`).
//! Spoiler: every single one becomes `0`. That failure is the whole reason
//! Chapter 2 (`naive`) exists — it introduces a *scale* to stretch the
//! tiny weight values into the i8 range before casting.

fn quantize_simple(x: f32) -> i8 {
    x as i8 // truncates toward zero, saturates at the i8 boundary
}

fn dequantize_simple(q: i8) -> f32 {
    q as f32
}

fn main() {
    let weights = [0.42_f32, -0.10, 0.70, -0.50, 0.99, -0.99];

    println!("{:>8}  {:>4}  {:>8}", "input", "→i8", "back→f32");
    println!("{:>8}  {:>4}  {:>8}", "-----", "----", "--------");
    for &w in &weights {
        let q = quantize_simple(w);
        let back = dequantize_simple(q);
        println!("{w:>8.2}  {q:>4}  {back:>8.2}");
    }

    println!("\nevery weight collapsed to 0. Chapter 2 (`naive`) fixes this");
    println!("by introducing a *scale* before the cast.");
}
