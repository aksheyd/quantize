//! # Chapter 8 — alternating (learned codes and scale)
//!
//! **Previously** (`ch07_learned`): we froze chapter 5's codes and fit the
//! line by least squares, which cut this block's MSE by about a third.
//!
//! **Problem**: the codes still come from min/max. After the fit moves the
//! line, a value may sit closer to a neighboring code, yet it keeps its own.
//!
//! **Fix**: alternate. Round every value to its nearest code on the current
//! line, refit the line to those codes, and repeat until no code changes.
//! Rounding picks the best codes for the line and the fit picks the best line
//! for the codes, so neither step can raise the MSE.
//!
//! **Still wrong**: it minimizes the error in the weights themselves. What
//! matters is the error in what the model computes with them, where a weight
//! that meets large inputs counts for more. GPTQ and AWQ minimize that error,
//! measured on sample inputs called *calibration data*.
//!
//! Run it: `cargo run --release --example ch08_alternating`

// 4-bit codes, as in chapters 5 and 7.
const SMALLEST_CODE: f32 = -8.0;
const LARGEST_CODE: f32 = 7.0;

/// Chapter 5: the minimum lands on the smallest code, the maximum on the largest.
fn asymmetric_params(values: &[f32]) -> (f32, f32) {
    let lowest = values.iter().copied().fold(f32::INFINITY, f32::min);
    let highest = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let scale = (highest - lowest) / (LARGEST_CODE - SMALLEST_CODE);
    (scale, SMALLEST_CODE - lowest / scale)
}

/// Chapter 7: the least-squares line through the (code, value) pairs.
fn fit_scale_and_zero_point(values: &[f32], codes: &[i32]) -> (f32, f32) {
    let count = values.len() as f32;
    let mut sum_codes = 0.0;
    let mut sum_values = 0.0;
    let mut sum_code_squared = 0.0;
    let mut sum_code_times_value = 0.0;
    for (&value, &code) in values.iter().zip(codes) {
        let code = code as f32;
        sum_codes += code;
        sum_values += value;
        sum_code_squared += code * code;
        sum_code_times_value += code * value;
    }
    let mean_code = sum_codes / count;
    let mean_value = sum_values / count;
    let code_spread = sum_code_squared - sum_codes * mean_code;
    let scale = (sum_code_times_value - sum_codes * mean_value) / code_spread;
    let offset = mean_value - scale * mean_code;
    (scale, -offset / scale)
}

/// Each value's nearest code on the line `scale * (code - zero_point)`.
fn round_to_codes(values: &[f32], (scale, zero_point): (f32, f32)) -> Vec<i32> {
    let mut codes = Vec::new();
    for &value in values {
        let code = (value / scale + zero_point).round();
        codes.push(code.clamp(SMALLEST_CODE, LARGEST_CODE) as i32);
    }
    codes
}

/// Mean squared error of the codes decoded on the line.
fn mse(values: &[f32], codes: &[i32], (scale, zero_point): (f32, f32)) -> f32 {
    let mut total = 0.0;
    for (&value, &code) in values.iter().zip(codes) {
        let gap = value - scale * (code as f32 - zero_point);
        total += gap * gap;
    }
    total / values.len() as f32
}

fn report(step: &str, values: &[f32], codes: &[i32], line: (f32, f32)) {
    let error = mse(values, codes, line);
    println!("{step:<8} codes {codes:?}  mse={error:.6}");
}

fn main() {
    // Chapter 7's block: seven small values and one outlier.
    let values = [0.02_f32, -0.09, 0.10, 0.03, -0.04, 0.13, 0.04, -0.90];

    let mut line = asymmetric_params(&values);
    let mut codes = round_to_codes(&values, line);
    report("min/max", &values, &codes, line);
    loop {
        line = fit_scale_and_zero_point(&values, &codes);
        report("fit", &values, &codes, line);
        let new_codes = round_to_codes(&values, line);
        if new_codes == codes {
            break;
        }
        codes = new_codes;
        report("round", &values, &codes, line);
    }

    println!("\nNo code wants to move, so we stop. The first two lines are chapters 5 and 7;");
    println!("rounding again moved one value to a better code, and the MSE kept falling.");
    println!("On typical blocks of 32 weights it falls by about 14%, not this block's 73%.");
    println!("The library stops at chapter 7, and this is where the tutorial stops for now.");
}
