//! Saves the tiny llama's quantized tensors, loads them back, and generates
//! from them, for schemes that get a gguf file and one that doesn't.

use std::fs;

use quantize::{Scheme, f16};
use quantize_files::{gguf, safetensors};

use super::{encode, temporary_directory, tiny_llama};
use crate::checkpoint::quantize_matrices;
use crate::generate::generate;
use crate::model::Model;
use crate::saved_files::SavedFiles;

/// Each scheme, with the gguf file it saves, if any, and its safetensors
/// file. ggml's blocks have no zero-point, so asymmetric ones get no gguf.
const SCHEMES: [(Scheme, Option<&str>, &str); 3] = [
    (
        Scheme::Q4_32,
        Some("smollm-135m-q4_0.gguf"),
        "smollm-135m-q4_32.safetensors",
    ),
    (
        Scheme::Q8_32,
        Some("smollm-135m-q8_0.gguf"),
        "smollm-135m-q8_32.safetensors",
    ),
    (
        Scheme::Asymmetric { bits: 4, block: 32 },
        None,
        "smollm-135m-asymmetric-q4_32.safetensors",
    ),
];

#[test]
fn saved_files_hold_the_tensors_saved() {
    let tiny = tiny_llama("saved-tensors");
    let directory = temporary_directory("saved-tensors-files");
    for (scheme, gguf_name, safetensors_name) in SCHEMES {
        let tensors = quantize_matrices(&tiny.tensors, scheme).unwrap();
        let saved = SavedFiles::save(&directory, scheme, &tensors).unwrap();
        assert_eq!(saved.gguf, gguf_name.map(|name| directory.join(name)));
        assert_eq!(saved.safetensors, directory.join(safetensors_name));

        // assert! rather than assert_eq!, which would print every value of
        // both models when they differ.
        let from_safetensors = safetensors::read::<f16>(&saved.safetensors).unwrap();
        assert!(from_safetensors == tensors, "{scheme}'s safetensors file");
        if let Some(gguf_path) = &saved.gguf {
            let (_metadata, from_gguf) = gguf::read(gguf_path).unwrap();
            assert!(from_gguf == tensors, "{scheme}'s gguf file");
        }
    }
    fs::remove_dir_all(&directory).unwrap();
}

#[test]
fn saved_files_generate_the_tokens_that_memory_does() {
    let tiny = tiny_llama("saved-generation");
    let directory = temporary_directory("saved-generation-files");
    // The tiny llama answers this prompt with "which" a few times, then
    // "later" over and over, and each of the three schemes turns at a
    // different token. Most changes to a matrix move where one of them
    // turns; the test above, which compares the tensors, catches the rest.
    let prompt = encode(&tiny.tokenizer, "the first school in the city");
    let generate_from = |tensors| {
        let model = Model::new(&tiny.config, tensors).unwrap();
        let mut tokens = Vec::new();
        generate(&model, &prompt, 32, |token| tokens.push(token)).unwrap();
        tokens
    };
    for (scheme, _, _) in SCHEMES {
        let tensors = quantize_matrices(&tiny.tensors, scheme).unwrap();
        let saved = SavedFiles::save(&directory, scheme, &tensors).unwrap();
        let from_memory = generate_from(tensors);
        assert_eq!(from_memory.len(), 32);

        // load() reads the gguf file when there is one, as main does.
        assert_eq!(
            generate_from(saved.load().unwrap()),
            from_memory,
            "{scheme}"
        );
        let from_safetensors = safetensors::read(&saved.safetensors).unwrap();
        assert_eq!(generate_from(from_safetensors), from_memory, "{scheme}");
    }
    fs::remove_dir_all(&directory).unwrap();
}
