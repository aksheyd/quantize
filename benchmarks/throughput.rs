//! Quantize and dequantize speed against candle, timed the same way in both
//! libraries by `speed.rs`, plus this crate's fused `dot` and `matmul`.
//! `matmul` multiplies the weights by 16 vectors, so its time covers all 16.
//!
//! Run: `cargo run --release -p benchmarks --example throughput`

mod speed;

use half::f16;
use quantize::{Scheme, quantize};
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

    // Decode an adaptive matrix one row at a time, the way an embedding table
    // is read.
    let adaptive = Scheme::Adaptive {
        block: 32,
        tolerance: 0.1,
    };
    let mut table = adaptive.quantize::<f16>(&values).unwrap();
    table.set_shape(SIDE, SIDE).unwrap();
    let rows = time_per_value(|| {
        let mut decoded = vec![0.0; SIDE * SIDE];
        for row in 0..SIDE {
            let out = &mut decoded[row * SIDE..(row + 1) * SIDE];
            table.dequantize_row_into(row, out).unwrap();
        }
        decoded
    });
    println!("{:<18}{rows:>10.3}", "adaptive rows");
    Ok(())
}
