//! Quantize and dequantize speed against candle, timed the same way in both
//! libraries by `speed.rs`, plus this crate's fused `dot` and `matmul`.
//! `matmul` multiplies 4-bit and 8-bit weights by 16 vectors, so its time
//! covers all 16, and then by one, as a language model does for each token it
//! generates.
//!
//! Run: `cargo run --release -p benchmarks --example throughput`

mod speed;

use half::f16;
use quantize::Scheme;
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

    let dot_schemes = [
        ("8-bit dot", Scheme::Q8_32),
        ("5-bit dot", Scheme::Symmetric { bits: 5, block: 32 }),
        ("asymmetric dot", Scheme::Asymmetric { bits: 4, block: 32 }),
        (
            "adaptive dot",
            Scheme::Adaptive {
                block: 32,
                tolerance: 0.1,
            },
        ),
    ];
    for (name, scheme) in dot_schemes {
        let quantized = scheme.quantize::<f16>(&values).unwrap();
        let dot = time_per_value(|| quantized.dot(&values).unwrap());
        println!("{name:<18}{dot:>10.3}");
    }

    let sixteen_vectors = &values[..16 * SIDE];
    let one_vector = &values[..SIDE];
    for (name, scheme) in [("4-bit", Scheme::Q4_32), ("8-bit", Scheme::Q8_32)] {
        let mut matrix = scheme.quantize::<f16>(&values).unwrap();
        matrix.set_shape(SIDE, SIDE).unwrap();
        let matmul = time_per_value(|| matrix.matmul(sixteen_vectors).unwrap());
        println!("{:<18}{matmul:>10.3}", format!("{name} matmul ×16"));
        let matmul = time_per_value(|| matrix.matmul(one_vector).unwrap());
        println!("{:<18}{matmul:>10.3}", format!("{name} matmul ×1"));
    }

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
