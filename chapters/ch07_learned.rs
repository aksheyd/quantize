//! # Chapter 7 — learned scale and zero-point
//!
//! **Previously** (`ch06_adaptive`): bit width follows a tolerance, but
//! scale/zero-point still come from min/max of the block.
//!
//! **Problem**: min/max fit the *range*, not the *error*. Outliers set the
//! scale; the rest of the block pays for it.
//!
//! **Fix**: start from chapter 5's codes, freeze them, and treat dequant as a
//! line: `value ≈ scale * code + offset`, with `offset = -scale * zero_point`.
//! Fit one line per block by *least squares*: it picks the line with the
//! smallest *mean squared error* (MSE), the average squared gap between each
//! value and what comes back. Min/max is just one such line, so for the same
//! codes the fit never does worse.
//!
//! **Still wrong**: codes are frozen. Learning the codes together with the
//! scale is the next step.
//!
//! Run it: `cargo run --release --example ch07_learned`

// 4-bit codes, as in chapter 5.
const SMALLEST_CODE: f32 = -8.0;
const LARGEST_CODE: f32 = 7.0;

/// Chapter 5: the minimum lands on the smallest code, the maximum on the largest.
fn asymmetric_params(values: &[f32]) -> (f32, f32) {
    let lowest = values.iter().copied().fold(f32::INFINITY, f32::min);
    let highest = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let scale = (highest - lowest) / (LARGEST_CODE - SMALLEST_CODE);
    (scale, SMALLEST_CODE - lowest / scale)
}

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

fn decode(codes: &[i32], (scale, zero_point): (f32, f32)) -> Vec<f32> {
    codes
        .iter()
        .map(|&code| scale * (code as f32 - zero_point))
        .collect()
}

/// Mean squared error: the average of each squared gap.
fn mse(predicted: &[f32], expected: &[f32]) -> f32 {
    predicted
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        / predicted.len() as f32
}

fn main() {
    // Seven small values and one outlier. Min/max stretches the codes to reach
    // the outlier, so the small values share the top few codes.
    let values = [0.02_f32, -0.09, 0.10, 0.03, -0.04, 0.13, 0.04, -0.90];

    let (scale, zero_point) = asymmetric_params(&values);
    let codes: Vec<i32> = values
        .iter()
        .map(|&value| (value / scale + zero_point).round() as i32)
        .collect();
    let min_max_back = decode(&codes, (scale, zero_point));

    let (fitted_scale, fitted_zero_point) = fit_scale_and_zero_point(&values, &codes);
    let fitted_back = decode(&codes, (fitted_scale, fitted_zero_point));

    println!("codes    {codes:?}");
    println!(
        "min/max  scale={scale:.5} zero_point={zero_point:.3}  mse={:.6}",
        mse(&min_max_back, &values)
    );
    println!(
        "fitted   scale={fitted_scale:.5} zero_point={fitted_zero_point:.3}  mse={:.6}",
        mse(&fitted_back, &values)
    );
    println!("\nDequant is a line. Fit the line; keep the codes. The fit cuts the MSE");
    println!("by about a third, but it can't move a value to a better code.");
}
