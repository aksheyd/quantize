//! # Chapter 5 — asymmetric, precision-aware
//!
//! **Previously** (`ch04_block`): per-block scales aren't symmetric around zero.
//!
//! **Problem**:  So, if all values are positive, we waste the negative half of
//!  the quantized range. For example, `[0.1, 0.3, 0.7, 1.1]` at 8 bits with a
//! `0.01` would quantize to `[10, 30, 70, 110]` -> no negative values used.
//!
//! **Fix**: Store a zero-point per block that fits real [min, max]. We "stretch"
//! the quantized range to fit the real range and "shift" to center the real zero.
//!
//! **Still wrong**: we are leaving optimizations on the table by using a fixed bit width for all blocks.
//! For example, a block with small ranges of values doesn't need 16 bits to achieve the same precision.
//!
//! Run it: `cargo run --release --example ch05_asymmetric`

const fn max_int(b: u32) -> i32 {
    (1_i32 << (b - 1)) - 1
}
const fn min_int(b: u32) -> i32 {
    -(1_i32 << (b - 1))
}

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

fn asym_params(block: &[f32], bits: u32) -> (f32, f32) {
    let rmin = block.iter().copied().fold(f32::INFINITY, f32::min);
    let rmax = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if rmin >= rmax {
        return (1.0, 0.0);
    }
    let qmin = min_int(bits) as f32;
    let scale = (rmax - rmin) / (max_int(bits) as f32 - qmin);
    let zp = qmin - rmin / scale;
    (scale, zp)
}

fn q_asym(x: f32, scale: f32, zp: f32, bits: u32) -> i32 {
    ((x / scale + zp).round() as i32).clamp(min_int(bits), max_int(bits))
}
fn dq_asym(q: i32, scale: f32, zp: f32) -> f32 {
    (q as f32 - zp) * scale
}

fn main() {
    let tol = 0.001_f32;
    let tiny = [0.500, 0.501, 0.499, 0.5005];
    let wide = [0.10, 0.30, 0.70, 1.10];

    let bt = choose_bits(0.5005 - 0.499, tol);
    let (st, zpt) = asym_params(&tiny, bt);
    let ct: Vec<_> = tiny.iter().map(|&x| q_asym(x, st, zpt, bt)).collect();
    let recon_t: Vec<_> = ct.iter().map(|&q| dq_asym(q, st, zpt)).collect();

    let bw = choose_bits(1.10 - 0.10, tol);
    let (sw, zpw) = asym_params(&wide, bw);
    let cw: Vec<_> = wide.iter().map(|&x| q_asym(x, sw, zpw, bw)).collect();
    let recon_w: Vec<_> = cw.iter().map(|&q| dq_asym(q, sw, zpw)).collect();

    println!("tiny bits={} recon={:?}", bt, recon_t);
    println!("wide bits={} recon={:?}", bw, recon_w);
    println!("Asymmetric centers grid per block; precision-aware picks bits by range.");
}
