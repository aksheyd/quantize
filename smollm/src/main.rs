//! # smollm
//!
//! Quantize [SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M),
//! a small llama, save it as the files that models ship in, and generate
//! text from the saved file next to the f32 model it came from:
//!
//! 1. Download the model's config, tokenizer, and weights from Hugging Face,
//!    or read them from `--model DIR`.
//! 2. Quantize every weight matrix with `--scheme`, `Q4_32` by default, with
//!    f16 scales. The norms' weights stay f32.
//! 3. Save the quantized model in `target/smollm`, or `--output DIR`:
//!    as `smollm-135m-q4_0.gguf`, llama.cpp's format, which llama.cpp runs,
//!    when ggml has blocks for the scheme, and as
//!    `smollm-135m-q4_32.safetensors`, Hugging Face's format, for any
//!    scheme.
//! 4. Load the quantized model back from the gguf file, or from the
//!    safetensors file when the scheme has no gguf.
//! 5. Generate 32 tokens greedily from the f32 model and the loaded one,
//!    printing the text as it comes, then how fast each one went.
//!
//! The model is plain Rust. Its files read in order: `config.rs`,
//! `linear.rs`, `attention.rs`, `layer.rs`, `model.rs`, `weights.rs`, and
//! `generate.rs`. `saved_files.rs` saves the quantized model and loads it
//! back, and the files in `saved_files/` write the gguf as llama.cpp's own
//! converter does.
//!
//! Run: `cargo run --release -p smollm -- the capital of france is`, and add
//! `--scheme Q8_32` for 8-bit weights.

mod arguments;
mod attention;
mod checkpoint;
mod config;
mod generate;
mod layer;
mod linear;
mod model;
mod printer;
mod saved_files;
mod weights;

#[cfg(test)]
mod tests;

use std::error::Error;

use arguments::Arguments;
use checkpoint::{Checkpoint, MODEL_ID, quantize_matrices};
use generate::generate;
use model::Model;
use printer::TextPrinter;
use saved_files::SavedFiles;

const NEW_TOKENS: usize = 32;

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = Arguments::parse(std::env::args().skip(1))?;
    let checkpoint = match &arguments.model_directory {
        Some(directory) => Checkpoint::read(directory)?,
        None => {
            println!("loading {MODEL_ID}");
            Checkpoint::download()?
        }
    };
    let Checkpoint {
        config,
        tokenizer,
        tensors,
    } = checkpoint;

    let quantized_tensors = quantize_matrices(&tensors, arguments.scheme)?;
    println!(
        "quantized the weight matrices with {} and f16 scales",
        arguments.scheme
    );
    let saved = SavedFiles::save(
        &arguments.output_directory,
        arguments.scheme,
        &config,
        &tokenizer,
        &quantized_tensors,
    )?;
    // The quantized model comes from its file, not from the tensors in
    // memory, so its text shows that the file holds the whole model.
    let models = [
        ("f32", Model::new(&config, tensors)?),
        ("quantized", Model::new(&config, saved.load(&config)?)?),
    ];

    let encoding = tokenizer.encode(arguments.prompt.as_str(), true);
    let prompt = encoding.map_err(|error| format!("couldn't tokenize the prompt: {error}"))?;
    let prompt = prompt.get_ids().to_vec();
    let mut speeds = Vec::new();
    for (name, model) in &models {
        print!("\n{name}: ");
        let mut text = TextPrinter::new(&tokenizer, &prompt);
        let elapsed = generate(model, &prompt, NEW_TOKENS, |token| text.push(token))?;
        println!();
        speeds.push((name, NEW_TOKENS as f64 / elapsed.as_secs_f64()));
    }
    println!();
    for (name, tokens_per_second) in speeds {
        println!("{name}: {tokens_per_second:.1} tokens per second");
    }
    Ok(())
}
