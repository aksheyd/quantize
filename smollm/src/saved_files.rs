//! Save the quantized model as the files that models ship in, then load it
//! back, so the quantized text comes from a file that other programs can
//! read too:
//!
//! - gguf, llama.cpp's format, holds `Q4_32` and `Q8_32` matrices as ggml's
//!   `Q4_0` and `Q8_0` blocks: `smollm-135m-q4_0.gguf` for `Q4_32`. ggml has
//!   no blocks for other schemes, so they get no gguf file. llama.cpp runs
//!   the file as it runs its own conversion of SmolLM-135M, which takes
//!   three changes: llama.cpp's names for the tensors, in
//!   `llama_cpp_names.rs`; its order for the rows of the query and key
//!   matrices, in `rope_order.rs`; and metadata that describes the model, in
//!   `gguf_metadata.rs`, and its tokenizer, in `tokenizer_metadata.rs`.
//! - safetensors, Hugging Face's format, holds a matrix of any scheme as the
//!   bytes that quantize writes, by Hugging Face's names:
//!   `smollm-135m-q4_32.safetensors` for `Q4_32`.
//!
//! Each file holds every tensor, the quantized matrices and the f32 norms,
//! so either one loads back alone.

mod gguf_metadata;
mod llama_cpp_names;
pub mod rope_order;
mod tokenizer_metadata;

use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use quantize::{Scheme, f16};
use quantize_files::{Tensor, gguf, safetensors};
use tokenizers::Tokenizer;

use crate::config::Config;

/// The quantized model's files.
pub struct SavedFiles {
    /// The gguf file, if ggml has blocks for the scheme.
    pub gguf: Option<PathBuf>,
    /// The safetensors file, which every scheme gets.
    pub safetensors: PathBuf,
}

impl SavedFiles {
    /// Save `tensors`, quantized with `scheme`, into `directory`, replacing
    /// any files of the same names, and print each file's path and size.
    /// The gguf file also holds what llama.cpp needs to run the model, from
    /// `config` and `tokenizer`.
    pub fn save(
        directory: &Path,
        scheme: Scheme,
        config: &Config,
        tokenizer: &Tokenizer,
        tensors: &BTreeMap<String, Tensor<f16>>,
    ) -> Result<Self, Box<dyn Error>> {
        fs::create_dir_all(directory)
            .map_err(|error| format!("{}: {error}", directory.display()))?;
        let gguf_path = match ggml_blocks_for(scheme) {
            Some((blocks, file_type)) => {
                let path = directory.join(format!("smollm-135m-{blocks}.gguf"));
                let metadata = gguf_metadata::metadata(config, tokenizer, file_type)?;
                let tensors = llama_cpp_names::to_llama_cpp(tensors, config)?;
                gguf::write(&path, &metadata, &tensors)?;
                print_saved(&path)?;
                Some(path)
            }
            None => None,
        };
        let scheme_name = scheme_in_file_name(scheme);
        let safetensors_path = directory.join(format!("smollm-135m-{scheme_name}.safetensors"));
        safetensors::write(&safetensors_path, tensors)?;
        print_saved(&safetensors_path)?;
        Ok(Self {
            gguf: gguf_path,
            safetensors: safetensors_path,
        })
    }

    /// Load the tensors back from the gguf file, the one llama.cpp reads, or
    /// from the safetensors file when the scheme has no gguf. Either way,
    /// they come back by Hugging Face's names, in Hugging Face's order, as
    /// [`Model::new`](crate::model::Model::new) takes them.
    pub fn load(&self, config: &Config) -> Result<BTreeMap<String, Tensor<f16>>, Box<dyn Error>> {
        let path = self.gguf.as_ref().unwrap_or(&self.safetensors);
        println!("loading the quantized model from {}", path.display());
        if self.gguf.is_some() {
            let (_metadata, tensors) = gguf::read(path)?;
            Ok(llama_cpp_names::to_hugging_face(tensors, config)?)
        } else {
            Ok(safetensors::read(path)?)
        }
    }
}

/// The ggml blocks that hold `scheme`'s matrices, or `None` if no ggml
/// blocks hold them as they are: their name, as llama.cpp's file names give
/// it, and llama.h's number for a file whose matrices are all those blocks.
///
/// ggml's `Q4_0` and `Q8_0` are quantize's symmetric 4-bit and 8-bit blocks
/// of 32, with the same f16 scales, laid out another way. ggml's blocks
/// have no place for an asymmetric block's zero-point, or for an adaptive
/// block's width of its own.
fn ggml_blocks_for(scheme: Scheme) -> Option<(&'static str, u32)> {
    match scheme {
        Scheme::Symmetric { bits: 4, block: 32 } => Some(("q4_0", 2)),
        Scheme::Symmetric { bits: 8, block: 32 } => Some(("q8_0", 7)),
        _ => None,
    }
}

/// `scheme` without the spaces and parentheses its text has, so a file name
/// can hold it: `q4_32` for symmetric 4-bit blocks of 32, as quantize names
/// [`Scheme::Q4_32`], `asymmetric-q4_32` for asymmetric ones, and
/// `adaptive-32-0.002` for adaptive blocks of 32 with a tolerance of 0.002.
fn scheme_in_file_name(scheme: Scheme) -> String {
    match scheme {
        Scheme::Symmetric { bits, block } => format!("q{bits}_{block}"),
        Scheme::Asymmetric { bits, block } => format!("asymmetric-q{bits}_{block}"),
        Scheme::Adaptive { block, tolerance } => format!("adaptive-{block}-{tolerance}"),
    }
}

/// Print where a file was saved, and its size.
fn print_saved(path: &Path) -> Result<(), String> {
    let metadata = fs::metadata(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let bytes = metadata.len() as f64;
    let size = if bytes < 1e6 {
        format!("{:.1} KB", bytes / 1e3)
    } else {
        format!("{:.1} MB", bytes / 1e6)
    };
    println!("saved {} ({size})", path.display());
    Ok(())
}
