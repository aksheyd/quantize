//! Quantize and dequantize speed against candle, timed the same way in both
//! libraries by `speed.rs`, plus this crate's fused `dot` and `matmul`.
//! `matmul` multiplies the weights by 16 vectors, so its time covers all 16.
//!
//! Run: `cargo run --release --example throughput`

mod speed;

use half::f16;
use quantize::quantize;
use speed::{ITERATIONS, SIDE, time_per_value};

fn main() -> candle_core::Result<()> {
    let values = speed::values();
    println!("{SIDE}x{SIDE} values, f16 scales, median of {ITERATIONS} calls, in ns per value");
    println!(
        "the quant and dequant rows are each the fastest of {} such medians\n",
        speed::PASSES
    );
    println!("{:<18}{:>10}{:>10}", "kernel", "quantize", "candle");
    println!("{:-<38}", "");
    for kernel in speed::measure(&values)? {
        println!(
            "{:<18}{:>10.3}{:>10.3}",
            kernel.name, kernel.this_crate, kernel.candle
        );
    }

    let quantized_8bit = quantize::<f16, 8, 32>(&values).unwrap();
    let mut quantized_4bit = quantize::<f16, 4, 32>(&values).unwrap();
    quantized_4bit.set_shape(SIDE, SIDE).unwrap();
    let sixteen_vectors = &values[..16 * SIDE];
    let dot = time_per_value(|| quantized_8bit.dot(&values).unwrap());
    let matmul = time_per_value(|| quantized_4bit.matmul(sixteen_vectors).unwrap());
    println!("{:<18}{dot:>10.3}", "8-bit dot");
    println!("{:<18}{matmul:>10.3}", "4-bit matmul ×16");
    Ok(())
}
