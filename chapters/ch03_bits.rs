//! # Chapter 3 — bits (any bit width)
//!
//! **Previously** (`ch02_naive`): we introduced a scale so the full i8 range
//! is used. Round-trips look great.
//!
//! **Problem**: we are hardcoded to 8 bits. Sometimes you want 4-bit
//! (smaller model) or 16-bit (higher fidelity).
//!
//! **Fix**: parameterize the bit width with a const generic `BITS`. The
//! algorithm is identical — only the smallest and largest code change, and the
//! scale now divides by the new largest code. Keep `BITS` between 2 and 16,
//! like the library does: at 1 bit the largest code is 0, so there is nothing
//! to scale to. The codes sit in `i32`s to keep things simple; a real format
//! packs them tightly, two 4-bit codes to a byte.
//!
//! **Still wrong**: one outlier in a million-element tensor wrecks the scale.
//!
//! Run it: `cargo run --release --example ch03_bits`

const fn largest_code<const BITS: u32>() -> i32 {
    (1_i32 << (BITS - 1)) - 1
}

const fn smallest_code<const BITS: u32>() -> i32 {
    -(1_i32 << (BITS - 1))
}

fn quantize_bits<const BITS: u32>(x: f32, scale: f32) -> i32 {
    let smallest = smallest_code::<BITS>() as f32;
    let largest = largest_code::<BITS>() as f32;
    (x / scale).round().clamp(smallest, largest) as i32
}

fn dequantize_bits(code: i32, scale: f32) -> f32 {
    code as f32 * scale
}

fn choose_scale_bits<const BITS: u32>(values: &[f32]) -> f32 {
    let max_abs = values.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    if max_abs > 0.0 {
        max_abs / largest_code::<BITS>() as f32
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

fn roundtrip<const BITS: u32>(weights: &[f32]) {
    let scale = choose_scale_bits::<BITS>(weights);
    println!("\n--- {BITS}-bit  (scale = {scale:.8}) ---");
    println!("{:>8}  {:>6}  {:>10}", "input", "code", "back");
    println!("{:>8}  {:>6}  {:>10}", "-----", "----", "--------");
    let mut reconstructed = Vec::new();
    for &w in weights {
        let code = quantize_bits::<BITS>(w, scale);
        let back = dequantize_bits(code, scale);
        println!("{w:>8.2}  {code:>6}  {back:>10.4}");
        reconstructed.push(back);
    }
    println!("worst error: {:.6}", worst_error(weights, &reconstructed));
}

fn main() {
    let weights = [0.42_f32, -0.10, 0.70, -0.50, 0.99, -0.99];

    roundtrip::<4>(&weights);
    roundtrip::<8>(&weights);
    roundtrip::<16>(&weights);

    println!("\nSame algorithm, different precision. Chapter 4 (`ch04_block`) shows");
    println!("why a single per-tensor scale is still not enough and introduces blocks.");
}
