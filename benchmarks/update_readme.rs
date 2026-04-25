//! Run the harness and rewrite the comparison table in README.md.
//!
//! Usage: `cargo run --release --example update_readme`

mod harness;

use harness::Harness;
use std::fs;

const START: &str = "<!-- comparison:start -->";
const END: &str = "<!-- comparison:end -->";
const MATRIX_SIZE: usize = 128;
const RUNS: usize = 10;

fn main() -> candle_core::Result<()> {
    let report = Harness::new(MATRIX_SIZE, RUNS)?.run()?;

    let mut rows = String::from("| method | bits/elt | mse (mean ± std) | cosine (mean ± std) |\n");
    rows.push_str("| --- | ---: | ---: | ---: |\n");
    for m in &report.methods {
        rows.push_str(&format!(
            "| {} | {:.1} | {:.6} ± {:.6} | {:.6} ± {:.6} |\n",
            m.name,
            m.bits_per_element,
            m.stats.mse_mean,
            m.stats.mse_std,
            m.stats.cosine_mean,
            m.stats.cosine_std,
        ));
    }
    let table = format!(
        "{START}\n\n{rows}\n_matrix size: {n}x{n}, runs: {runs}_\n\n{END}",
        n = report.matrix_size,
        runs = report.runs,
    );

    let readme = fs::read_to_string("README.md").expect("README.md not found");
    let start = readme.find(START).expect("missing start marker");
    let end = readme.find(END).expect("missing end marker") + END.len();
    let updated = format!("{}{table}{}", &readme[..start], &readme[end..]);
    fs::write("README.md", updated).expect("failed to write README.md");

    println!("updated README.md");
    Ok(())
}
