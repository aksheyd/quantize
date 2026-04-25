//! Run the harness and rewrite the comparison table in README.md.
//!
//! Usage: `cargo run --release --example update_readme`

mod harness;

use harness::Harness;
use std::fs;

const START: &str = "<!-- comparison:start -->";
const END: &str = "<!-- comparison:end -->";

fn main() -> candle_core::Result<()> {
    let report = Harness::new(128)?.run()?;

    let table = format!(
        "{START}\n\n\
         | method | mse | cosine |\n\
         | --- | ---: | ---: |\n\
         | quantize | {:.6} | {:.6} |\n\
         | [candle q8_0](https://github.com/huggingface/candle) | {:.6} | {:.6} |\n\n\
        _matrix size: {}x{}_\n\n\
         {END}",
        report.quantize.mse,
        report.quantize.cosine,
        report.candle_q8_0.mse,
        report.candle_q8_0.cosine,
        report.matrix_size,
        report.matrix_size,
    );

    let readme = fs::read_to_string("README.md").expect("README.md not found");
    let start = readme.find(START).expect("missing start marker");
    let end = readme.find(END).expect("missing end marker") + END.len();
    let updated = format!("{}{table}{}", &readme[..start], &readme[end..]);
    fs::write("README.md", updated).expect("failed to write README.md");

    println!("updated README.md");
    Ok(())
}
