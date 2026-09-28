//! # Chapter 6 — adaptive mixed precision
//!
//! **Previously** (`ch05_asymmetric`): a zero-point stretches the codes to
//! `[min, max]`, but every block still uses the same bit width.
//!
//! **Problem**: a nearly-constant block does not need 4 bits. Paying 4 bits
//! for a 0.002 range buys more precision than it needs, while a busy block may
//! need more than 4.
//!
//! **Fix**: pick a *tolerance* by hand: the worst error we'll accept. Rounding
//! to the nearest code is off by at most half a step, so each block gets the
//! fewest bits whose half-step is `<= tolerance`. Quiet blocks drop to 2–3
//! bits; busy blocks get up to 8, which may not be enough.
//!
//! **Still wrong**: scale and zero-point are computed from min/max, which caps
//! the worst error but ignores the average one. They can be *learned*.
//!
//! Run it: `cargo run --release --example ch06_adaptive`

// `bits` is a plain argument now instead of a const generic like `BITS` in
// chapters 3–5, because each block picks its own while the program runs.
fn largest_code(bits: u32) -> i32 {
    (1_i32 << (bits - 1)) - 1
}
fn smallest_code(bits: u32) -> i32 {
    -(1_i32 << (bits - 1))
}

/// Fewest bits in `2..=8` whose half-step is `<= tolerance`, or else 8.
fn choose_bits(range: f32, tolerance: f32) -> u32 {
    if range <= 0.0 {
        return 2;
    }
    for bits in 2..=8 {
        let steps = (largest_code(bits) - smallest_code(bits)) as f32;
        let half_step = range / steps / 2.0;
        if half_step <= tolerance {
            return bits;
        }
    }
    8
}

/// Asymmetric quantization at `bits`: to codes and back.
fn roundtrip(block: &[f32], bits: u32) -> Vec<f32> {
    let lowest = block.iter().copied().fold(f32::INFINITY, f32::min);
    let highest = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let smallest = smallest_code(bits) as f32;
    let largest = largest_code(bits) as f32;
    // A flat block has no range to stretch; with a scale of 1 its one value
    // still lands exactly on the smallest code.
    let scale = if highest > lowest {
        (highest - lowest) / (largest - smallest)
    } else {
        1.0
    };
    let zero_point = smallest - lowest / scale;
    let mut back = Vec::new();
    for &x in block {
        let code = (x / scale + zero_point).round().clamp(smallest, largest) as i32;
        back.push((code as f32 - zero_point) * scale);
    }
    back
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
    let tolerance = 0.001_f32;
    let tensor = [
        0.500, 0.501, 0.499, 0.5005, // quiet
        0.10, 0.33, 0.71, 1.10, // busy
    ];

    println!("tolerance = {tolerance}\n");
    println!("block   range  bits  worst error  within tolerance");
    println!("-----   -----  ----  -----------  ----------------");
    for (i, block) in tensor.chunks(4).enumerate() {
        let lowest = block.iter().copied().fold(f32::INFINITY, f32::min);
        let highest = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = highest - lowest;
        let bits = choose_bits(range, tolerance);
        let error = worst_error(block, &roundtrip(block, bits));
        let within = if error <= tolerance { "yes" } else { "no" };
        println!("{i:>5}  {range:>6.4}  {bits:>4}  {error:>11.5}  {within}");
    }

    println!("\nSame tensor, two precisions: the quiet block needs only 2 bits, while the");
    println!("busy block gets 8, the most we allow, and still misses the tolerance. The");
    println!("library does this in `quantize::adaptive::quantize`, but a miss is an error.");
    println!("Chapter 7 (`ch07_learned`) fits scale and zero-point to the values instead.");
}
