//! # Chapter 5 — asymmetric (zero-point)
//!
//! **Previously** (`ch04_block`): each block got its own scale from its largest
//! magnitude, so its codes cover `-max..=max`, symmetric around zero. The data
//! in a block may not be.
//!
//! **Problem**: So, if all values are positive, we waste the negative half of
//! the quantized range. For example, `[0.10, 0.33, 0.71, 1.10]` at 8 bits with
//! a scale of `0.01` would quantize to `[10, 33, 71, 110]` -> no negative codes.
//!
//! **Fix**: Store a *zero-point* per block: where real 0 sits on the code line.
//! We "stretch" the quantized range to fit the block's real `[min, max]` and
//! "shift" it so the minimum lands on the smallest code and the maximum on the
//! largest. Chapter 4's symmetric scale is the case where the zero-point is 0.
//! Here it's a float like the scale, so it can land between codes or far
//! outside them. Many libraries store it as a code, so 0.0 comes back exactly.
//! As a float, it costs as much to store as the scale: an f32 of each per 4
//! values makes 4 + 64/4 = 20 bits per value; real formats go from 4.5 to 5.
//!
//! **Still wrong**: every block gets the same bit width, however wide its
//! range. At 4 bits the quiet block below comes back far more precisely than
//! the wide one.
//!
//! Run it: `cargo run --release --example ch05_asymmetric`

const fn largest_code<const BITS: u32>() -> i32 {
    (1_i32 << (BITS - 1)) - 1
}
const fn smallest_code<const BITS: u32>() -> i32 {
    -(1_i32 << (BITS - 1))
}

/// Chapter 4: the largest magnitude lands on the largest code.
fn symmetric_params<const BITS: u32>(block: &[f32]) -> (f32, f32) {
    let max_abs = block.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    if max_abs > 0.0 {
        (max_abs / largest_code::<BITS>() as f32, 0.0)
    } else {
        (1.0, 0.0)
    }
}

/// The minimum lands on the smallest code, the maximum on the largest.
fn asymmetric_params<const BITS: u32>(block: &[f32]) -> (f32, f32) {
    let lowest = block.iter().copied().fold(f32::INFINITY, f32::min);
    let highest = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if lowest >= highest {
        // A flat block has no range to stretch, so use the symmetric scale.
        return symmetric_params::<BITS>(block);
    }
    let steps = (largest_code::<BITS>() - smallest_code::<BITS>()) as f32;
    let scale = (highest - lowest) / steps;
    let zero_point = smallest_code::<BITS>() as f32 - lowest / scale;
    (scale, zero_point)
}

/// Quantize each value to a code, then decode the code back.
fn roundtrip<const BITS: u32>(block: &[f32], (scale, zero_point): (f32, f32)) -> Vec<f32> {
    let smallest = smallest_code::<BITS>() as f32;
    let largest = largest_code::<BITS>() as f32;
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

fn compare<const BITS: u32>(name: &str, block: &[f32]) {
    let symmetric = roundtrip::<BITS>(block, symmetric_params::<BITS>(block));
    let (scale, zero_point) = asymmetric_params::<BITS>(block);
    let asymmetric = roundtrip::<BITS>(block, (scale, zero_point));

    println!("{name} block, {BITS} bits");
    println!("      value  symmetric  asymmetric");
    for (i, value) in block.iter().enumerate() {
        println!("{value:>11.4}{:>11.4}{:>12.4}", symmetric[i], asymmetric[i]);
    }
    let symmetric_error = worst_error(block, &symmetric);
    let asymmetric_error = worst_error(block, &asymmetric);
    println!("worst error{symmetric_error:>11.4}{asymmetric_error:>12.4}");
    println!("zero-point {:>11.2}{zero_point:>12.2}\n", 0.0);
}

fn main() {
    compare::<4>("quiet", &[0.500, 0.501, 0.499, 0.5005]);
    compare::<4>("wide", &[0.10, 0.33, 0.71, 1.10]);

    println!("On the quiet block, symmetric puts every value on the same code; a");
    println!("zero-point spreads the codes over each block's own range instead. The");
    println!("library does this in `quantize::asymmetric::quantize`. Both blocks got");
    println!("4 bits, though, and the quiet one came back far more precisely. Chapter 6");
    println!("(`ch06_adaptive`) picks the bit width per block.");
}
