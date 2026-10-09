//! A checkpoint: the three files a model ships as, from Hugging Face or a
//! directory, and the tensors inside them.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::Path;

use hf_hub::api::sync::ApiBuilder;
use quantize::{Scheme, f16};
use quantize_files::{Tensor, safetensors};
use tokenizers::Tokenizer;

use crate::config::Config;

pub const MODEL_ID: &str = "HuggingFaceTB/SmolLM-135M";

pub struct Checkpoint {
    /// The model's sizes, from `config.json`.
    pub config: Config,
    /// What turns text into tokens and back, from `tokenizer.json`.
    pub tokenizer: Tokenizer,
    /// Every weight, by name, from `model.safetensors`. A file may store
    /// them as f32, f16, or bf16, and they all read as f32.
    pub tensors: BTreeMap<String, Tensor<f16>>,
}

impl Checkpoint {
    /// Download SmolLM-135M from Hugging Face, or reuse the copy an earlier
    /// run cached.
    pub fn download() -> Result<Self, Box<dyn Error>> {
        // Unlike Api::new(), from_env() reads HF_HOME (where the cache lives)
        // and HF_ENDPOINT (which server to download from).
        let repository = ApiBuilder::from_env().build()?.model(MODEL_ID.to_string());
        // ureq's errors don't say which URL failed, so each error starts with it.
        let download = |file: &str| {
            repository
                .get(file)
                .map_err(|error| format!("{}: {error}", repository.url(file)))
        };
        Self::read_files(
            &download("config.json")?,
            &download("tokenizer.json")?,
            &download("model.safetensors")?,
        )
    }

    /// Read the same three files from `directory`.
    pub fn read(directory: &Path) -> Result<Self, Box<dyn Error>> {
        let file = |name: &str| directory.join(name);
        Self::read_files(
            &file("config.json"),
            &file("tokenizer.json"),
            &file("model.safetensors"),
        )
    }

    fn read_files(config: &Path, tokenizer: &Path, weights: &Path) -> Result<Self, Box<dyn Error>> {
        let config = Config::read(config)?;
        let tokenizer = Tokenizer::from_file(tokenizer)
            .map_err(|error| format!("{}: {error}", tokenizer.display()))?;
        // Each token looks up its row of the embedding table, so the table
        // needs a row for every token the tokenizer can make.
        let largest_token = tokenizer.get_vocab(true).into_values().max().unwrap_or(0);
        if largest_token as usize >= config.vocab_size {
            return Err(format!(
                "the tokenizer makes token {largest_token}, but the config's vocab_size is {}",
                config.vocab_size
            )
            .into());
        }
        let tensors = safetensors::read(weights)?;
        Ok(Self {
            config,
            tokenizer,
            tensors,
        })
    }
}

/// Quantize every matrix in `tensors` with `scheme`, with f16 scales. The
/// vectors, the norms' weights, stay f32: they hold a tiny share of the
/// model's values, so quantizing them would save almost nothing.
pub fn quantize_matrices(
    tensors: &BTreeMap<String, Tensor<f16>>,
    scheme: Scheme,
) -> Result<BTreeMap<String, Tensor<f16>>, quantize::Error> {
    let mut quantized = BTreeMap::new();
    for (name, tensor) in tensors {
        let tensor = match tensor {
            Tensor::Float { shape, values } if shape.len() == 2 => {
                let mut weights = scheme.quantize(values)?;
                weights.set_shape(shape[0], shape[1])?;
                Tensor::Quantized(weights)
            }
            other => other.clone(),
        };
        quantized.insert(name.clone(), tensor);
    }
    Ok(quantized)
}
