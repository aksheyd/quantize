//! Quantization comparison playground.
//!
//! Runs `RUNS` fresh matmuls for every method registered in
//! `harness/methods.rs` and prints mean MSE.

mod harness;
use harness::{Comparison, Harness};

const MATRIX_SIZE: usize = 1024;
const RUNS: usize = 50;

fn main() -> candle_core::Result<()> {
    let report = Harness::new(MATRIX_SIZE, RUNS)?.run()?;
    print_report(&report);
    Ok(())
}

fn print_report(r: &Comparison) {
    println!("matrix_size = {MATRIX_SIZE}, runs = {RUNS}\n");
    println!("{:<12}{:>14}", "bits/value", "mse");
    println!("{:-<12}{:->14}", "", "");
    for m in &r.methods {
        println!("{:<12.1}{:>14.6}", m.bits_per_element, m.mse);
    }
    println!("\nbits/value = storage for one number, including its scale.");
}
