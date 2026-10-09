//! llama.cpp's names for a llama's tensors, which it loads a gguf file's
//! tensors by. Hugging Face names a layer's tensors like
//! `model.layers.0.self_attn.q_proj.weight`, and llama.cpp like
//! `blk.0.attn_q.weight`. SmolLM's output head is its embedding table, so
//! the file has no `output.weight`, and llama.cpp scores the next token with
//! `token_embd.weight`, as `model.rs` does.

use std::collections::BTreeMap;

use quantize::f16;
use quantize_files::Tensor;

use super::rope_order;
use crate::config::Config;

/// The tensors outside the layers: Hugging Face's name, then llama.cpp's.
const MODEL_TENSORS: [(&str, &str); 2] = [
    ("model.embed_tokens.weight", "token_embd.weight"),
    ("model.norm.weight", "output_norm.weight"),
];

/// Each layer's tensors: Hugging Face's name after `model.layers.N.`, then
/// llama.cpp's after `blk.N.`.
const LAYER_TENSORS: [(&str, &str); 9] = [
    ("input_layernorm.weight", "attn_norm.weight"),
    ("self_attn.q_proj.weight", "attn_q.weight"),
    ("self_attn.k_proj.weight", "attn_k.weight"),
    ("self_attn.v_proj.weight", "attn_v.weight"),
    ("self_attn.o_proj.weight", "attn_output.weight"),
    ("post_attention_layernorm.weight", "ffn_norm.weight"),
    ("mlp.gate_proj.weight", "ffn_gate.weight"),
    ("mlp.up_proj.weight", "ffn_up.weight"),
    ("mlp.down_proj.weight", "ffn_down.weight"),
];

/// Hugging Face's tensors as llama.cpp's converter writes them: by
/// llama.cpp's names, with q's and k's rows in llama.cpp's order.
pub fn to_llama_cpp(
    tensors: &BTreeMap<String, Tensor<f16>>,
    config: &Config,
) -> Result<BTreeMap<String, Tensor<f16>>, String> {
    let mut renamed = BTreeMap::new();
    for (hugging_face, llama_cpp) in names(config.layer_count) {
        let Some(tensor) = tensors.get(&hugging_face) else {
            return Err(format!("the model has no {hugging_face}"));
        };
        let tensor = match rope_heads(&llama_cpp, config) {
            Some(heads) => rope_order::to_llama_cpp(tensor.clone(), heads)
                .map_err(|error| format!("{hugging_face}: {error}"))?,
            None => tensor.clone(),
        };
        renamed.insert(llama_cpp, tensor);
    }
    if renamed.len() < tensors.len() {
        return Err("llama.cpp has no name for some of the model's tensors".into());
    }
    Ok(renamed)
}

/// The tensors of a gguf file that [`to_llama_cpp`] wrote, by Hugging Face's
/// names again, with q's and k's rows back in Hugging Face's order.
pub fn to_hugging_face(
    mut tensors: BTreeMap<String, Tensor<f16>>,
    config: &Config,
) -> Result<BTreeMap<String, Tensor<f16>>, String> {
    let mut renamed = BTreeMap::new();
    for (hugging_face, llama_cpp) in names(config.layer_count) {
        let Some(tensor) = tensors.remove(&llama_cpp) else {
            return Err(format!("the gguf file has no {llama_cpp}"));
        };
        let tensor = match rope_heads(&llama_cpp, config) {
            Some(heads) => rope_order::to_hugging_face(tensor, heads)
                .map_err(|error| format!("{llama_cpp}: {error}"))?,
            None => tensor,
        };
        renamed.insert(hugging_face, tensor);
    }
    Ok(renamed)
}

/// Each tensor's Hugging Face name, then llama.cpp's, in a llama of
/// `layer_count` layers.
fn names(layer_count: usize) -> Vec<(String, String)> {
    let mut names: Vec<_> = MODEL_TENSORS
        .iter()
        .map(|(hugging_face, llama_cpp)| (hugging_face.to_string(), llama_cpp.to_string()))
        .collect();
    for layer in 0..layer_count {
        names.extend(LAYER_TENSORS.iter().map(|(hugging_face, llama_cpp)| {
            let hugging_face = format!("model.layers.{layer}.{hugging_face}");
            (hugging_face, format!("blk.{layer}.{llama_cpp}"))
        }));
    }
    names
}

/// How many heads rope turns in the tensor llama.cpp calls `name`: q's rows
/// hold `head_count` heads and k's `key_value_head_count`. Rope turns
/// queries and keys alone.
fn rope_heads(name: &str, config: &Config) -> Option<usize> {
    if name.ends_with(".attn_q.weight") {
        Some(config.head_count)
    } else if name.ends_with(".attn_k.weight") {
        Some(config.key_value_head_count)
    } else {
        None
    }
}
