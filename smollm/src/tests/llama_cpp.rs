//! Checks that the gguf file has the metadata and the tensor names that
//! llama.cpp looks up to run it.

use std::fs;

use quantize::Scheme;
use quantize_files::gguf::{self, Value};

use super::{temporary_directory, tiny_llama};
use crate::checkpoint::quantize_matrices;
use crate::saved_files::SavedFiles;

/// The keys of llama.cpp's own conversion of SmolLM-135M, but for
/// `general.finetune`, which SmolLM-135M has none of, and
/// `tokenizer.ggml.unknown_token_id`, which byte-level BPE never uses.
const KEYS: &str = "general.architecture general.type general.name general.basename \
    general.size_label general.file_type general.quantization_version llama.vocab_size \
    llama.context_length llama.embedding_length llama.feed_forward_length llama.block_count \
    llama.attention.head_count llama.attention.head_count_kv llama.attention.key_length \
    llama.attention.value_length llama.rope.dimension_count llama.rope.freq_base \
    llama.attention.layer_norm_rms_epsilon tokenizer.ggml.model tokenizer.ggml.pre \
    tokenizer.ggml.add_space_prefix tokenizer.ggml.add_bos_token tokenizer.ggml.tokens \
    tokenizer.ggml.token_type tokenizer.ggml.merges tokenizer.ggml.bos_token_id \
    tokenizer.ggml.eos_token_id";

/// Each layer's tensors, by llama.cpp's names after `blk.N.`.
const LAYER_TENSORS: &str = "attn_norm attn_q attn_k attn_v attn_output \
    ffn_norm ffn_gate ffn_up ffn_down";

#[test]
fn gguf_file_has_what_llama_cpp_needs() {
    let tiny = tiny_llama("llama-cpp");
    let directory = temporary_directory("llama-cpp-files");
    let tensors = quantize_matrices(&tiny.tensors, Scheme::Q4_32).unwrap();
    let (config, tokenizer) = (&tiny.config, &tiny.tokenizer);
    let saved = SavedFiles::save(&directory, Scheme::Q4_32, config, tokenizer, &tensors).unwrap();
    let (metadata, tensors) = gguf::read(saved.gguf.unwrap()).unwrap();
    fs::remove_dir_all(&directory).unwrap();

    let mut keys: Vec<&str> = KEYS.split_whitespace().collect();
    keys.sort();
    assert!(metadata.keys().eq(keys), "{:?}", metadata.keys());
    let text = |text: &str| Value::String(text.to_string());
    assert_eq!(metadata["general.architecture"], text("llama"));
    assert_eq!(metadata["general.file_type"], Value::U32(2));
    assert_eq!(metadata["llama.block_count"], Value::U32(2));
    assert_eq!(metadata["llama.attention.head_count_kv"], Value::U32(2));
    assert_eq!(metadata["llama.rope.dimension_count"], Value::U32(16));
    assert_eq!(metadata["tokenizer.ggml.model"], text("gpt2"));
    assert_eq!(metadata["tokenizer.ggml.pre"], text("smollm"));

    // A token for each row of the embedding table: the special tokens, a
    // token for each byte, the merged tokens, then rows of no token.
    let (Value::Array(tokens), Value::Array(types), Value::Array(merges)) = (
        &metadata["tokenizer.ggml.tokens"],
        &metadata["tokenizer.ggml.token_type"],
        &metadata["tokenizer.ggml.merges"],
    ) else {
        panic!("the tokens, their types, and the merges are arrays");
    };
    let last = config.vocab_size - 1;
    assert_eq!((tokens.len(), types.len()), (last + 1, last + 1));
    assert_eq!(tokens[0], text("<|endoftext|>"));
    assert_eq!(tokens[3 + usize::from(b' ')], text("Ġ"));
    assert_eq!(tokens[last], text(&format!("[PAD{last}]")));
    assert_eq!(types[..4], [3, 3, 3, 1].map(Value::I32));
    assert_eq!(types[last], Value::I32(5));
    assert_eq!(merges[..3], [text("Ġ t"), text("Ġt h"), text("Ġth e")]);

    // The embedding table is the output head too, so there's no
    // `output.weight`.
    let mut names = vec!["token_embd.weight".to_string(), "output_norm.weight".into()];
    for layer in 0..config.layer_count {
        let parts = LAYER_TENSORS.split_whitespace();
        names.extend(parts.map(|part| format!("blk.{layer}.{part}.weight")));
    }
    names.sort();
    assert!(tensors.keys().eq(&names), "{:?}", tensors.keys());
}
