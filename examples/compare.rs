//! Quantization learning playground.
//!
//! Edit `quantize` and `dequantize` below, then `cargo run --release` to see
//! how close you got to f32 ground truth and Candle's Q8_0 quantization.
//! Harness/setup code lives in `harness/`; printing happens here.

mod harness;
use harness::{Comparison, Harness};

use quantize::{dequantize, quantize};

fn main() -> candle_core::Result<()> {
    // Run a 128x128 matmul and calculate error compared
    // to the f32 ground truth and Candle Q8_0 algorithm.
    let report = Harness::new(128)?.run(quantize, dequantize)?;
    print_report(&report);
    Ok(())
}

fn print_report(report: &Comparison) {
    println!("matrix_size = {}\n", report.matrix_size);
    println!("{:<14}{:>12}{:>12}", "method", "mse", "cosine");
    println!("{:<14}{:>12}{:>12}", "------", "---", "------");
    println!(
        "{:<14}{:>12.6}{:>12.6}",
        "yours", report.quantize.mse, report.quantize.cosine
    );
    println!(
        "{:<14}{:>12.6}{:>12.6}",
        "candle Q8_0", report.candle_q8_0.mse, report.candle_q8_0.cosine
    );
}
