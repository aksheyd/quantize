//! # Chapter 4 — block (per-block scales)
//!
//! **Previously** (`ch03_bits`): we allowed for any bit precision, but
//! we were still left with one scale for the whole tensor.
//!
//! **Problem**: One outlier forces a huge scale. One large or tiny value in a
//! million-element tensor can mess up the quantization scale for everyone else.
//!
//! **Fix**: Split the tensor into fixed-size blocks and compute an independent
//! scale per block.
//!
//! **Still wrong**: One block's range can be wastefully skewed. Only using the
//! positive or negative half of the quantized range is a common failure mode.
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
    (x / scale)
        .round()
        .clamp(min_int::<BITS>() as f32, max_int::<BITS>() as f32) as i32
}

fn compare<const BITS: u32, const BLOCK: usize>(weights: &[f32]) {
    // Per-tensor (Ch. 3 style)
    let global_scale = choose_scale::<BITS>(weights);
    let global_codes: Vec<i32> = weights
        .iter()
        .map(|&x| quantize_val::<BITS>(x, global_scale))
        .collect();

    // Per-block
    let scales: Vec<f32> = weights.chunks(BLOCK).map(choose_scale::<BITS>).collect();
    let block_codes: Vec<i32> = weights
        .chunks(BLOCK)
        .zip(&scales)
        .flat_map(|(b, &s)| b.iter().map(move |&x| quantize_val::<BITS>(x, s)))
        .collect();

    println!(
        "weights: {weights:?}\n\nglobal scale: {:.4}\nblock scales: {:.4?}\n\n",
        global_scale, scales
    );

    println!("{:>3}  {:>6}  {:>6}  {:>6}", "i", "w", "global", "blocks");
    println!("{:-<3}  {:-<6}  {:-<6}  {:-<6}", "", "", "", "");
    for (i, ((&w, &g), &b)) in weights
        .iter()
        .zip(&global_codes)
        .zip(&block_codes)
        .enumerate()
    {
        println!("{:>3}  {:>6.2}  {:>6}  {:>6}", i, w, g, b);
    }

    println!("\nGlobal scale is dominated by the outlier → first block collapses to 0.");
    println!("Per-block scales rescue the small values while still handling the spike.");
}

fn main() {
    let weights = [0.04_f32, 0.05, -0.03, 0.06, 4.0, 0.10, -0.05, 0.08];

    compare::<4, 4>(&weights);
}
