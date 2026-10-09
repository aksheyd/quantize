//! Save the quantized model as the files that models ship in, then load it
//! back, so the quantized text comes from a file that other programs can
//! read too:
//!
//! - gguf, llama.cpp's format, holds `Q4_32` and `Q8_32` matrices as ggml's
//!   `Q4_0` and `Q8_0` blocks: `smollm-135m-q4_0.gguf` for `Q4_32`. ggml has
//!   no blocks for other schemes, so they get no gguf file.
//! - safetensors, Hugging Face's format, holds a matrix of any scheme as the
//!   bytes that quantize writes: `smollm-135m-q4_32.safetensors` for `Q4_32`.
//!
//! Each file holds every tensor, the quantized matrices and the f32 norms,
//! by its Hugging Face name, so either one loads back alone.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use quantize::{Scheme, f16};
use quantize_files::{Tensor, gguf, safetensors};

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
    pub fn save(
        directory: &Path,
        scheme: Scheme,
        tensors: &BTreeMap<String, Tensor<f16>>,
    ) -> Result<Self, Box<dyn Error>> {
        fs::create_dir_all(directory)
            .map_err(|error| format!("{}: {error}", directory.display()))?;
        let gguf_path = match ggml_blocks_for(scheme) {
            Some(blocks) => {
                let path = directory.join(format!("smollm-135m-{blocks}.gguf"));
                // llama.cpp runs a file whose metadata names its architecture,
                // and looks its tensors up by llama.cpp's own names. These
                // are Hugging Face's, so the metadata stays empty, and
                // llama.cpp won't take the file for a model it can run.
                gguf::write(&path, &gguf::Metadata::new(), tensors)?;
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
    /// from the safetensors file when the scheme has no gguf.
    pub fn load(&self) -> Result<BTreeMap<String, Tensor<f16>>, quantize_files::Error> {
        let path = self.gguf.as_ref().unwrap_or(&self.safetensors);
        println!("loading the quantized model from {}", path.display());
        if self.gguf.is_some() {
            gguf::read(path).map(|(_metadata, tensors)| tensors)
        } else {
            safetensors::read(path)
        }
    }
}

/// The name of the ggml blocks that hold `scheme`'s matrices, as llama.cpp's
/// file names give it, or `None` if no ggml blocks hold them as they are.
///
/// ggml's `Q4_0` and `Q8_0` are quantize's symmetric 4-bit and 8-bit blocks
/// of 32, with the same f16 scales, laid out another way. ggml's blocks
/// have no place for an asymmetric block's zero-point, or for an adaptive
/// block's width of its own.
fn ggml_blocks_for(scheme: Scheme) -> Option<&'static str> {
    match scheme {
        Scheme::Symmetric { bits: 4, block: 32 } => Some("q4_0"),
        Scheme::Symmetric { bits: 8, block: 32 } => Some("q8_0"),
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
