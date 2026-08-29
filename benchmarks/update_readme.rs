//! Run the harness and rewrite the quality table in README.md.
//!
//! Usage: `cargo run --release --example update_readme`

mod harness;

use harness::Harness;
use std::fs;

const START: &str = "<!-- comparison:start -->";
const END: &str = "<!-- comparison:end -->";
const MATRIX_SIZE: usize = 1024;
const RUNS: usize = 50;

fn main() -> candle_core::Result<()> {
    let report = Harness::new(MATRIX_SIZE, RUNS)?.run()?;

    let mut rows = String::from("| bits/value | quantize mse | candle mse |\n");
    rows.push_str("| ---: | ---: | ---: |\n");
    for pair in report.methods.chunks(2) {
        let Some([q, c]) = pair.get(..2) else { break };
        rows.push_str(&format!(
            "| {:.1} | {:.6} | {:.6} |\n",
            q.bits_per_element, q.mse, c.mse,
        ));
    }
    let table = format!("{START}\n\n{rows}\n{END}");

    let readme = fs::read_to_string("README.md").expect("README.md not found");
    let start = readme.find(START).expect("missing start marker");
    let end = readme.find(END).expect("missing end marker") + END.len();
    let updated = format!("{}{table}{}", &readme[..start], &readme[end..]);
    fs::write("README.md", updated).expect("failed to write README.md");

    println!("updated README.md");
    Ok(())
}
