//! # Chapter 7 — learned scale and zero-point
//!
//! **Previously** (`ch06_adaptive`): bit width follows a tolerance, but
//! scale/zero-point still come from min/max of the block.
//!
//! **Problem**: min/max fit the *range*, not the *error*. Outliers set the
//! scale; the rest of the block pays for it.
//!
//! **Fix**: freeze the integer codes and treat dequant as a line:
//! `value ≈ scale * code + offset`, with `offset = -scale * zero_point`.
//! One best-fit line per block.
//!
//! **Still wrong**: codes are frozen. Jointly learning codes is the next
//! research step, not this chapter.
//!
//! Run it: `cargo run --release --example ch07_learned`

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

fn mse(predicted: &[f32], expected: &[f32]) -> f32 {
    predicted
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        / predicted.len() as f32
}

fn main() {
    // One outlier dominates a min/max scale; a best-fit line ignores it better.
    let values = [0.10_f32, 0.12, 0.11, 0.13, 4.0];
    let codes = [10, 12, 11, 13, 127];

    let max = values.iter().copied().fold(0.0_f32, f32::max);
    let minmax_scale = max / 127.0;
    let minmax_back: Vec<f32> = codes
        .iter()
        .map(|&code| code as f32 * minmax_scale)
        .collect();

    let (scale, zero_point) = fit_scale_and_zero_point(&values, &codes);
    let fitted: Vec<f32> = codes
        .iter()
        .map(|&code| scale * (code as f32 - zero_point))
        .collect();

    println!(
        "minmax  scale={minmax_scale:.5}  mse={:.5}",
        mse(&minmax_back, &values)
    );
    println!(
        "fitted  scale={scale:.5} zero_point={zero_point:.3}  mse={:.5}",
        mse(&fitted, &values)
    );
    println!("\nDequant is a line. Fit the line; keep the codes.");
}
