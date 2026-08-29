//! # Chapter 6 — adaptive mixed precision
//!
//! **Previously** (`ch05_asymmetric`): a zero-point stretches the grid to
//! `[min, max]`, but every block still uses the same bit width.
//!
//! **Problem**: a nearly-constant block does not need 8 bits. Paying 8 bits
//! for a 0.002 range wastes memory that a wild block actually needs.
//!
//! **Fix**: pick bits from a *calibrated tolerance*. The half-step of the
//! integer grid must stay `<= tol`. Quiet blocks drop to 2–3 bits; busy
//! blocks keep 8.
//!
//! **Still wrong**: scale and zero-point are computed from min/max, not from
//! the reconstruction error we actually care about. They can be *learned*.
//!
//! Run it: `cargo run --release --example ch06_adaptive`

fn choose_bits(range: f32, tol: f32) -> u32 {
    if range <= 0.0 {
        return 2;
    }
    for b in 2..=8 {
        if range / ((1u32 << b) - 1) as f32 / 2.0 <= tol {
            return b;
        }
    }
    8
}

fn main() {
    let tol = 0.001_f32;
    let tensor = [
        0.500, 0.501, 0.499, 0.5005, // quiet
        0.10, 0.30, 0.70, 1.10, // busy
    ];

    println!("tol = {tol}\n");
    println!("{:>8}  {:>6}  {:>5}", "range", "bits", "block");
    println!("{:>8}  {:>6}  {:>5}", "-----", "----", "-----");
    for (i, block) in tensor.chunks(4).enumerate() {
        let rmin = block.iter().copied().fold(f32::INFINITY, f32::min);
        let rmax = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let bits = choose_bits(rmax - rmin, tol);
        println!("{:>8.4}  {:>6}  {:>5}", rmax - rmin, bits, i);
    }

    println!("\nSame tensor, two precisions. Chapter 7 (`ch07_learned`) treats");
    println!("scale and zero-point as a 1-neuron layer and fits them.");
}
