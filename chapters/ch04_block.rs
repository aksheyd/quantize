//! # Chapter 4 — block (per-block scales)
//!
//! **Previously** (`ch03_bits`): we allowed for any bit precision, but
//! we were still left with one scale for the whole tensor.
//!
//! **Problem**: One outlier forces a huge scale. One large value in a
//! million-element tensor can mess up the quantization scale for everyone else.
//!
//! **Fix**: Split the tensor into fixed-size blocks and compute an independent
//! scale per block. An outlier then only wrecks its own block's scale. Each
//! block stores its scale, though: here an f32 per 4 values, 8 bits per value.
//! Real formats use a 16-bit scale per 32 values: 4 + 16/32 = 4.5 bits per value.
//!
//! **Still wrong**: One block's range can be wastefully skewed. Only using the
//! positive or negative half of the quantized range is a common failure mode.
//!
//! Run it: `cargo run --release --example ch04_block`

const fn largest_code<const BITS: u32>() -> i32 {
    (1_i32 << (BITS - 1)) - 1
}
const fn smallest_code<const BITS: u32>() -> i32 {
    -(1_i32 << (BITS - 1))
}

fn choose_scale<const BITS: u32>(values: &[f32]) -> f32 {
    let max_abs = values.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    if max_abs > 0.0 {
        max_abs / largest_code::<BITS>() as f32
    } else {
        1.0
    }
}

fn quantize_value<const BITS: u32>(x: f32, scale: f32) -> i32 {
    let smallest = smallest_code::<BITS>() as f32;
    let largest = largest_code::<BITS>() as f32;
    (x / scale).round().clamp(smallest, largest) as i32
}

/// The biggest gap between an input and what came back.
fn worst_error(inputs: &[f32], outputs: &[f32]) -> f32 {
    let mut worst = 0.0_f32;
    for (input, output) in inputs.iter().zip(outputs) {
        worst = worst.max((input - output).abs());
    }
    worst
}

fn compare<const BITS: u32, const BLOCK: usize>(weights: &[f32]) {
    // One scale for the whole tensor, as in chapter 3
    let global_scale = choose_scale::<BITS>(weights);
    let mut global_back = Vec::new();
    for &x in weights {
        let code = quantize_value::<BITS>(x, global_scale);
        global_back.push(code as f32 * global_scale);
    }

    // One scale per block
    let mut scales = Vec::new();
    let mut block_back = Vec::new();
    for block in weights.chunks(BLOCK) {
        let scale = choose_scale::<BITS>(block);
        scales.push(scale);
        for &x in block {
            let code = quantize_value::<BITS>(x, scale);
            block_back.push(code as f32 * scale);
        }
    }

    println!("weights: {weights:?}\n");
    println!("global scale: {global_scale:.4}\nblock scales: {scales:.4?}\n\n");

    println!("{:>3}  {:>6}  {:>8}  {:>8}", "i", "w", "global", "blocks");
    println!("{:-<3}  {:-<6}  {:-<8}  {:-<8}", "", "", "", "");
    for (i, &weight) in weights.iter().enumerate() {
        println!(
            "{:>3}  {:>6.2}  {:>8.4}  {:>8.4}",
            i, weight, global_back[i], block_back[i]
        );
    }
    let global_error = worst_error(weights, &global_back);
    let block_error = worst_error(weights, &block_back);
    println!("\nworst error:    global {global_error:.4}, blocks {block_error:.4}");
    // Every scale is an f32: 32 bits, shared by the values it covers.
    let global_bits = BITS as f32 + 32.0 / weights.len() as f32;
    let block_bits = BITS as f32 + 32.0 / BLOCK as f32;
    println!("bits per value: global {global_bits:.1}, blocks {block_bits:.1}");

    println!("\nGlobal scale is dominated by the outlier → first block collapses to 0.");
    println!("Per-block scales rescue it. The outlier still ruins its own block, so");
    println!("smaller blocks would limit the damage to fewer values, but store more scales.");
    println!("The library does this in `quantize::symmetric::quantize`.");
}

fn main() {
    let weights = [0.16_f32, 0.20, -0.11, 0.24, 4.0, 0.10, -0.05, 0.08];

    compare::<4, 4>(&weights);
}
