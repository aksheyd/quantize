//! # Chapter 3 — bits (any bit width)
//!
//! **Previously** (`ch02_naive`): we introduced a scale so the full i8 range
//! is used. Round-trips look great.
//! **Problem**: we are hardcoded to 8 bits. Sometimes you want 4-bit
//! (smaller model) or 16-bit (higher fidelity).
//! **Fix**: parameterize the bit width with a const generic `BITS`. The
//! algorithm is identical — only the clamp bounds change.
//! **Still wrong**: one outlier in a million-element tensor wrecks the
//! single per-tensor scale. Chapter 4 (`block`) fixes this by computing
//! a separate scale for each small block of elements.
//!
//! Run it: `cargo run --release --example ch03_bits`

const fn max_int<const BITS: u32>() -> i32 {
    (1_i32 << (BITS - 1)) - 1
}

const fn min_int<const BITS: u32>() -> i32 {
    -(1_i32 << (BITS - 1))
}

fn quantize_bits<const BITS: u32>(x: f32, scale: f32) -> i32 {
    (x / scale)
        .round()
        .clamp(min_int::<BITS>() as f32, max_int::<BITS>() as f32) as i32
}

fn dequantize_bits(q: i32, scale: f32) -> f32 {
    q as f32 * scale
}

fn choose_scale_bits<const BITS: u32>(values: &[f32]) -> f32 {
    let max_abs = values.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    if max_abs > 0.0 {
        max_abs / max_int::<BITS>() as f32
    } else {
        1.0
    }
}

fn roundtrip<const BITS: u32>(weights: &[f32]) {
    let scale = choose_scale_bits::<BITS>(weights);
    println!("\n--- {BITS}-bit  (scale = {scale:.6}) ---");
    println!("{:>8}  {:>6}  {:>10}", "input", "code", "back");
    println!("{:>8}  {:>6}  {:>10}", "-----", "----", "--------");
    for &w in weights {
        let q = quantize_bits::<BITS>(w, scale);
        let back = dequantize_bits(q, scale);
        println!("{w:>8.2}  {q:>6}  {back:>10.4}");
    }
}

fn main() {
    let weights = [0.42_f32, -0.10, 0.70, -0.50, 0.99, -0.99];

    roundtrip::<4>(&weights);
    roundtrip::<8>(&weights);
    roundtrip::<16>(&weights);

    println!("\nSame algorithm, different precision. Chapter 4 (`ch04_block`) shows");
    println!("why a single per-tensor scale is still not enough and introduces blocks.");
}
