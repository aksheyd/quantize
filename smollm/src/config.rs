//! The model's sizes, read from its `config.json`.

use std::error::Error;
use std::fs;
use std::path::Path;

use serde_json::Value;

/// A llama's sizes, by the names `config.json` gives them. SmolLM-135M's are
/// in parentheses.
#[derive(Clone, Copy)]
pub struct Config {
    /// How many values stand for each token between layers (576).
    pub hidden_size: usize,
    /// How many values the mlp works with in the middle (1536).
    pub intermediate_size: usize,
    /// How many tokens the model knows (49152).
    pub vocab_size: usize,
    /// How many layers run one after another (30).
    pub layer_count: usize,
    /// How many attention heads each layer has (9).
    pub head_count: usize,
    /// How many heads of keys and values those heads share (3).
    pub key_value_head_count: usize,
    /// The base of rope's rotation frequencies (10000).
    pub rope_theta: f32,
    /// Keeps rms norm from dividing by zero (0.00001).
    pub rms_norm_epsilon: f32,
}

impl Config {
    pub fn read(path: &Path) -> Result<Self, Box<dyn Error>> {
        let text =
            fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
        let json: Value = serde_json::from_str(&text)?;
        let number = |key| {
            json[key]
                .as_f64()
                .ok_or(format!("{}: no {key}", path.display()))
        };

        // The output head reuses the embedding table, so a model with a
        // separate `lm_head.weight` would score tokens with the wrong matrix.
        if json["tie_word_embeddings"] != true {
            return Err(format!("{}: needs tie_word_embeddings", path.display()).into());
        }
        Ok(Self {
            hidden_size: number("hidden_size")? as usize,
            intermediate_size: number("intermediate_size")? as usize,
            vocab_size: number("vocab_size")? as usize,
            layer_count: number("num_hidden_layers")? as usize,
            head_count: number("num_attention_heads")? as usize,
            key_value_head_count: number("num_key_value_heads")? as usize,
            rope_theta: number("rope_theta")? as f32,
            rms_norm_epsilon: number("rms_norm_eps")? as f32,
        })
    }

    /// How many values each head works with (64).
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.head_count
    }
}
