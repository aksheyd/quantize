//! Quantization comparison playground.
//!
//! Runs `RUNS` fresh matmuls for every method registered in
//! `harness/methods.rs` and prints MSE / cosine as mean ± std.

mod harness;
use harness::{Comparison, Harness};

const MATRIX_SIZE: usize = 128;
const RUNS: usize = 10;

fn main() -> candle_core::Result<()> {
    let report = Harness::new(MATRIX_SIZE, RUNS)?.run()?;
    print_report(&report);
    Ok(())
}

fn print_report(r: &Comparison) {
    println!("matrix_size = {}, runs = {}\n", r.matrix_size, r.runs);
    println!(
        "{:<14}{:>10}{:>22}{:>22}",
        "method", "bits/elt", "mse (mean ± std)", "cosine (mean ± std)",
    );
    println!("{:-<14}{:->10}{:->22}{:->22}", "", "", "", "");
    for m in &r.methods {
        let mse = format!("{:.6} ± {:.6}", m.stats.mse_mean, m.stats.mse_std);
        let cos = format!("{:.6} ± {:.6}", m.stats.cosine_mean, m.stats.cosine_std);
        println!(
            "{:<14}{:>10.1}{:>22}{:>22}",
            m.name, m.bits_per_element, mse, cos,
        );
    }
}
