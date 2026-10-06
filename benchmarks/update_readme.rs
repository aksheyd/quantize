//! Run the benchmarks and rewrite README.md's quality table, plus this
//! machine's rows of its speed table. Rows from other architectures stay,
//! so each platform's speed comes from a run on that platform.
//!
//! Usage: `cargo run --release -p benchmarks --example update_readme`

mod harness;
mod speed;

use harness::Harness;
use speed::KernelTime;
use std::fs;

const START: &str = "<!-- comparison:start -->";
const END: &str = "<!-- comparison:end -->";
const SPEED_START: &str = "<!-- speed:start -->";
const SPEED_END: &str = "<!-- speed:end -->";
const MATRIX_SIZE: usize = 1024;
const RUNS: usize = 50;

fn main() -> candle_core::Result<()> {
    // Time the kernels first, before the quality runs load every core.
    let kernels = speed::measure(&speed::values())?;
    let report = Harness::new(MATRIX_SIZE, RUNS)?.run()?;

    let mut rows = String::from("| bits/value | quantize mse | candle mse |\n");
    rows.push_str("| ---: | ---: | ---: |\n");
    for pair in report.methods.chunks(2) {
        let Some([q, c]) = pair.get(..2) else { break };
        assert!(q.name.starts_with("quantize") && c.name.starts_with("candle"));
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
    let updated = with_speed_rows(&updated, &kernels);
    fs::write("README.md", updated).expect("failed to write README.md");

    println!("updated README.md");
    Ok(())
}

/// Replaces this machine's two rows of the speed table, which are labeled by
/// architecture, like `| quantize, x86_64 |`.
fn with_speed_rows(readme: &str, kernels: &[KernelTime; 4]) -> String {
    let start = readme.find(SPEED_START).expect("missing speed:start");
    let end = readme.find(SPEED_END).expect("missing speed:end");
    let old_table = &readme[start..end];
    let names = kernels.map(|kernel| kernel.name).join(" | ");
    let columns = format!("| ns/value | {names} |");
    assert!(old_table.contains(&columns), "speed table needs {columns}");

    let arch = std::env::consts::ARCH;
    let quantize_label = format!("quantize, {arch}");
    let candle_label = format!("candle, {arch}");
    let quantize_row = row(&quantize_label, kernels.map(|kernel| kernel.this_crate));
    let candle_row = row(&candle_label, kernels.map(|kernel| kernel.candle));

    let mut new_table = String::new();
    for line in old_table.lines() {
        if line.starts_with(&format!("| {quantize_label} |")) {
            new_table.push_str(&quantize_row);
        } else if line.starts_with(&format!("| {candle_label} |")) {
            new_table.push_str(&candle_row);
        } else {
            new_table.push_str(line);
        }
        new_table.push('\n');
    }
    format!("{}{new_table}{}", &readme[..start], &readme[end..])
}

fn row(label: &str, times: [f64; 4]) -> String {
    let cells = times.map(|time| format!("{time:.2}")).join(" | ");
    format!("| {label} | {cells} |")
}
