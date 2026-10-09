//! Runs the example on a tiny random llama, shaped like SmolLM-135M but with
//! 2 layers of width 64 and a byte-level BPE tokenizer of a few hundred
//! tokens, which it writes to a temporary directory and reads back as
//! `--model DIR` does.
//!
//! Random weights write gibberish, so these check agreement, not what the
//! models say: `logits.rs`, that the quantized model's logits stay close to
//! the f32 model's, `saved_files.rs`, that the quantized model's files load
//! back as they were saved and generate the tokens it does in memory, and
//! `rope_order.rs` and `llama_cpp.rs`, that the gguf file has q's and k's
//! rows in llama.cpp's order, and the metadata and names llama.cpp needs.

mod llama_cpp;
mod logits;
mod rope_order;
mod saved_files;
mod tokenizer;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use quantize::f16;
use quantize_files::{Tensor, safetensors};
use serde_json::json;
use tokenizers::Tokenizer;

use crate::checkpoint::Checkpoint;

/// Words that the tokenizer makes one token each, after a space. It spells
/// any other text out of shorter tokens, down to single bytes.
const WORDS: &str = "the of and in to a was is for on as by with that at from his it an \
    were are which this be or has had first also its new their one after but who not they her \
    she two been other when there all during into school time may years more most only over \
    city some world would where later up";

/// Write the tiny llama's `config.json`, `tokenizer.json`, and
/// `model.safetensors` into `directory`.
fn write_tiny_llama(directory: &Path) {
    let tokenizer = tokenizer::tokenizer_json(WORDS);
    let token_count = tokenizer["model"]["vocab"].as_object().unwrap().len();
    // Rounded up to a multiple of 64, as many models round their embedding
    // tables, so the last few rows belong to no token.
    let vocab_size = token_count.next_multiple_of(64);
    let (hidden, intermediate, layers) = (64, 128, 2);
    // 4 heads of 16 values each, which share 2 heads of keys and values.
    let (heads, key_value_heads, head_dim) = (4, 2, 16);
    let config = json!({
        "hidden_size": hidden, "intermediate_size": intermediate, "vocab_size": vocab_size,
        "num_hidden_layers": layers, "num_attention_heads": heads,
        "num_key_value_heads": key_value_heads, "rope_theta": 10000.0, "rms_norm_eps": 1e-5,
        "tie_word_embeddings": true, "max_position_embeddings": 256,
        "bos_token_id": 0, "eos_token_id": 0,
    });
    fs::write(directory.join("config.json"), config.to_string()).unwrap();
    fs::write(directory.join("tokenizer.json"), tokenizer.to_string()).unwrap();

    let (query_width, key_value_width) = (heads * head_dim, key_value_heads * head_dim);
    let mut shapes = vec![
        (
            "model.embed_tokens.weight".to_string(),
            vec![vocab_size, hidden],
        ),
        ("model.norm.weight".to_string(), vec![hidden]),
    ];
    for layer in 0..layers {
        let name = |part: &str| format!("model.layers.{layer}.{part}.weight");
        shapes.extend([
            (name("self_attn.q_proj"), vec![query_width, hidden]),
            (name("self_attn.k_proj"), vec![key_value_width, hidden]),
            (name("self_attn.v_proj"), vec![key_value_width, hidden]),
            (name("self_attn.o_proj"), vec![hidden, query_width]),
            (name("mlp.gate_proj"), vec![intermediate, hidden]),
            (name("mlp.up_proj"), vec![intermediate, hidden]),
            (name("mlp.down_proj"), vec![hidden, intermediate]),
            (name("input_layernorm"), vec![hidden]),
            (name("post_attention_layernorm"), vec![hidden]),
        ]);
    }

    // Matrices get seeded random values in -0.1..0.1, and norms start at 1.
    let mut seed = 0x1234_5678_u32;
    let mut random = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed as f32 / u32::MAX as f32 - 0.5) * 0.2
    };
    let mut tensors = BTreeMap::new();
    for (name, shape) in shapes {
        let count = shape.iter().product();
        let values = if shape.len() == 1 {
            vec![1.0; count]
        } else {
            (0..count).map(|_| random()).collect()
        };
        tensors.insert(name, Tensor::<f16>::Float { shape, values });
    }
    safetensors::write(directory.join("model.safetensors"), &tensors).unwrap();
}

/// A new directory of its own for `name`, in the system's temporary
/// directory, so tests that run at once never share files.
fn temporary_directory(name: &str) -> PathBuf {
    let directory = std::env::temp_dir().join(format!("smollm-{}-{name}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    directory
}

/// Write the tiny llama into a directory of its own for `test`, and read it
/// back as `--model DIR` does.
fn tiny_llama(test: &str) -> Checkpoint {
    let directory = temporary_directory(test);
    write_tiny_llama(&directory);
    let checkpoint = Checkpoint::read(&directory).unwrap();
    fs::remove_dir_all(&directory).unwrap();
    checkpoint
}

fn encode(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    tokenizer.encode(text, true).unwrap().get_ids().to_vec()
}
