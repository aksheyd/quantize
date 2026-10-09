//! The metadata llama.cpp reads before a gguf file's tensors: what the file
//! holds, the llama's sizes, and its tokenizer, by the keys and with the
//! values that llama.cpp's converter, `convert_hf_to_gguf.py`, writes for
//! SmolLM-135M.

use quantize_files::gguf::{Metadata, Value};
use tokenizers::Tokenizer;

use super::tokenizer_metadata::tokenizer_metadata;
use crate::config::Config;

/// The metadata of a file whose matrices are ggml blocks of `file_type`.
pub fn metadata(
    config: &Config,
    tokenizer: &Tokenizer,
    file_type: u32,
) -> Result<Metadata, String> {
    let entries = general(file_type)
        .into_iter()
        .chain(sizes(config))
        .chain(tokenizer_metadata(config, tokenizer)?);
    Ok(entries.map(|(key, value)| (key.into(), value)).collect())
}

/// What the file holds. llama.cpp builds a llama from a file whose
/// architecture is `llama`, and reads its sizes under `llama.`.
fn general(file_type: u32) -> [(&'static str, Value); 7] {
    let text = |text: &str| Value::String(text.into());
    [
        ("general.architecture", text("llama")),
        ("general.type", text("model")),
        // The converter's names for SmolLM-135M, from its name on Hugging Face.
        ("general.name", text("SmolLM 135M")),
        ("general.basename", text("SmolLM")),
        ("general.size_label", text("135M")),
        // llama.h's number for a file of `Q4_0` matrices (2) or `Q8_0` ones
        // (7), with f32 norms.
        ("general.file_type", Value::U32(file_type)),
        // The layout of ggml's quantized blocks: 2, the one `Q4_0` and
        // `Q8_0` have had since ggml last changed it.
        ("general.quantization_version", Value::U32(2)),
    ]
}

/// The llama's sizes, from `config.json`.
fn sizes(config: &Config) -> [(&'static str, Value); 12] {
    let count = |count: usize| Value::U32(count as u32);
    let head_dim = count(config.head_dim());
    [
        ("llama.vocab_size", count(config.vocab_size)),
        ("llama.context_length", count(config.context_length)),
        ("llama.embedding_length", count(config.hidden_size)),
        ("llama.feed_forward_length", count(config.intermediate_size)),
        ("llama.block_count", count(config.layer_count)),
        ("llama.attention.head_count", count(config.head_count)),
        (
            "llama.attention.head_count_kv",
            count(config.key_value_head_count),
        ),
        // Each head's keys and values, and the values rope turns, are
        // head_dim wide.
        ("llama.attention.key_length", head_dim.clone()),
        ("llama.attention.value_length", head_dim.clone()),
        ("llama.rope.dimension_count", head_dim),
        ("llama.rope.freq_base", Value::F32(config.rope_theta)),
        (
            "llama.attention.layer_norm_rms_epsilon",
            Value::F32(config.rms_norm_epsilon),
        ),
    ]
}
