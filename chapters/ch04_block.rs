//! # Chapter 4 — block (per-block scales)
//!
//! **Previously** (`ch03_bits`): we allowed for any bit precision, but
//! we were still left with one scale for the whole tensor.
//!
//! **Problem**: One outlier forces a huge scale. One large value in a
//! million-element tensor can mess up the quantization scale for everyone else.
//!
//! **Fix**: Split the tensor into fixed-size blocks and compute an independent
//! scale per block.
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

fn compare<const BITS: u32, const BLOCK: usize>(weights: &[f32]) {
    // One scale for the whole tensor, as in chapter 3
    let global_scale = choose_scale::<BITS>(weights);
    let global_codes: Vec<i32> = weights
        .iter()
        .map(|&x| quantize_value::<BITS>(x, global_scale))
        .collect();

    // One scale per block
    let mut scales = Vec::new();
    let mut block_codes = Vec::new();
    for block in weights.chunks(BLOCK) {
        let scale = choose_scale::<BITS>(block);
        scales.push(scale);
        for &x in block {
            block_codes.push(quantize_value::<BITS>(x, scale));
        }
    }

    println!(
        "weights: {weights:?}\n\nglobal scale: {:.4}\nblock scales: {:.4?}\n\n",
        global_scale, scales
    );

    println!("{:>3}  {:>6}  {:>6}  {:>6}", "i", "w", "global", "blocks");
    println!("{:-<3}  {:-<6}  {:-<6}  {:-<6}", "", "", "", "");
    for (i, &weight) in weights.iter().enumerate() {
        println!(
            "{:>3}  {:>6.2}  {:>6}  {:>6}",
            i, weight, global_codes[i], block_codes[i]
        );
    }

    println!("\nGlobal scale is dominated by the outlier → first block collapses to 0.");
    println!("Per-block scales rescue the small values while still handling the spike.");
}

fn main() {
    let weights = [0.04_f32, 0.05, -0.03, 0.06, 4.0, 0.10, -0.05, 0.08];

    compare::<4, 4>(&weights);
}
