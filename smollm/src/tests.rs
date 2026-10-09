//! Runs the example on a tiny random llama, shaped like SmolLM-135M but with
//! 2 layers of width 64 and a 64-word tokenizer, which it writes to a
//! temporary directory and reads back as `--model DIR` does.
//!
//! Random weights write gibberish, so these check that the quantized model
//! agrees with the f32 one, not what either one says.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use quantize::{Scheme, f16};
use quantize_files::{Tensor, safetensors};
use serde_json::json;
use tokenizers::Tokenizer;

use crate::checkpoint::{Checkpoint, quantize_matrices};
use crate::generate::generate;
use crate::model::Model;

/// The tokenizer's 64 words. Any other word reads as `[UNK]`.
const WORDS: &str = "[UNK] the of and in to a was is for on as by with that at from his it an \
    were are which this be or has had first also its new their one after but who not they her \
    she two been other when there all during into school time may years more most only over \
    city some world would where later up";

/// Write the tiny llama's `config.json`, `tokenizer.json`, and
/// `model.safetensors` into `directory`.
fn write_tiny_llama(directory: &Path) {
    let (vocab_size, hidden, intermediate, layers) = (64, 64, 128, 2);
    // 4 heads of 16 values each, which share 2 heads of keys and values.
    let (heads, key_value_heads, head_dim) = (4, 2, 16);
    let config = json!({
        "hidden_size": hidden, "intermediate_size": intermediate, "vocab_size": vocab_size,
        "num_hidden_layers": layers, "num_attention_heads": heads,
        "num_key_value_heads": key_value_heads, "rope_theta": 10000.0, "rms_norm_eps": 1e-5,
        "tie_word_embeddings": true,
    });
    let vocab: BTreeMap<&str, usize> = WORDS.split_whitespace().zip(0..).collect();
    let tokenizer = json!({
        "normalizer": {"type": "Lowercase"},
        "pre_tokenizer": {"type": "Whitespace"},
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"},
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

/// Write the tiny llama into a directory of its own for `test`, and read it
/// back as `--model DIR` does.
fn tiny_llama(test: &str) -> Checkpoint {
    let directory = std::env::temp_dir().join(format!("smollm-{}-{test}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    write_tiny_llama(&directory);
    let checkpoint = Checkpoint::read(&directory).unwrap();
    fs::remove_dir_all(&directory).unwrap();
    checkpoint
}

fn encode(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    tokenizer.encode(text, true).unwrap().get_ids().to_vec()
}

/// The model's logits at every position of `tokens`.
fn logits(model: &Model, tokens: &[u32]) -> Vec<Vec<f32>> {
    let mut caches = model.new_caches();
    tokens
        .iter()
        .map(|&token| model.forward(token, &mut caches))
        .collect()
}

#[test]
fn quantized_logits_stay_close_to_f32s() {
    let tiny = tiny_llama("logits");
    let text = "the city was one of the first in the world and it has been a school since";
    let tokens = encode(&tiny.tokenizer, text);
    let float_logits = logits(
        &Model::new(&tiny.config, tiny.tensors.clone()).unwrap(),
        &tokens,
    );

    // The f32 logits reach about 1.4. Q8_32's stay within 0.017 of them, and
    // Q4_32's within 0.33. A transposed or reordered matrix moves them by
    // 0.29 or more, so Q8_32's tolerance catches one.
    for (scheme, tolerance) in [(Scheme::Q8_32, 0.05), (Scheme::Q4_32, 0.75)] {
        let quantized_tensors = quantize_matrices(&tiny.tensors, scheme).unwrap();
        let model = Model::new(&tiny.config, quantized_tensors).unwrap();
        let positions = float_logits.iter().zip(logits(&model, &tokens));
        for (position, (float, quantized)) in positions.enumerate() {
            let differences = float.iter().zip(&quantized).map(|(a, b)| (a - b).abs());
            let largest = differences.fold(0.0, f32::max);
            assert!(
                largest < tolerance,
                "{scheme} at position {position}: {largest}"
            );
        }
    }
}

#[test]
fn generation_is_deterministic() {
    let tiny = tiny_llama("deterministic");
    let prompt = encode(&tiny.tokenizer, "the first school in the city");
    let quantized_tensors = quantize_matrices(&tiny.tensors, Scheme::Q4_32).unwrap();
    let model = Model::new(&tiny.config, quantized_tensors).unwrap();
    let (mut first, mut second) = (Vec::new(), Vec::new());
    generate(&model, &prompt, 32, |token| first.push(token)).unwrap();
    generate(&model, &prompt, 32, |token| second.push(token)).unwrap();
    assert_eq!(first.len(), 32);
    assert_eq!(first, second);
}
