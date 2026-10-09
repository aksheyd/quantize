//! The tiny llama's tokenizer: byte-level BPE, as SmolLM's and GPT-2's are,
//! small enough to write out here, which llama.cpp runs as it runs SmolLM's.
//!
//! Byte-level BPE starts with a token for each of the 256 bytes, so it can
//! spell any text, then merges neighboring tokens into longer ones, in the
//! order its merges list them. Its vocabulary and merges are text, so each
//! byte has a printable character: a byte that prints as itself keeps it,
//! and the others, like the space, take the characters from 256 on, in
//! order, which makes the space `Ġ`.

use std::collections::BTreeMap;

use serde_json::{Value, json};

/// SmolLM's first three special tokens, at its ids for them.
const SPECIAL_TOKENS: [&str; 3] = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"];

/// `tokenizer.json` with SmolLM's first special tokens, a token for each
/// byte, and merges that make each of `words`, after a space, one token.
pub fn tokenizer_json(words: &str) -> Value {
    let byte_characters = byte_characters();
    let mut vocab: Vec<String> = SPECIAL_TOKENS.map(String::from).to_vec();
    vocab.extend(byte_characters.iter().map(char::to_string));
    let mut merges = Vec::new();
    for word in words.split_whitespace() {
        // " the" merges `Ġ` with `t`, then `Ġt` with `h`, then `Ġth` with `e`.
        let mut merged = byte_characters[usize::from(b' ')].to_string();
        for character in word.bytes().map(|byte| byte_characters[usize::from(byte)]) {
            let longer = format!("{merged}{character}");
            if !vocab.contains(&longer) {
                merges.push(format!("{merged} {character}"));
                vocab.push(longer.clone());
            }
            merged = longer;
        }
    }

    let ids: BTreeMap<&String, usize> = vocab.iter().zip(0..).collect();
    let added_tokens: Vec<Value> = (SPECIAL_TOKENS.iter().zip(0..))
        .map(|(content, id)| {
            json!({"id": id, "content": content, "special": true, "normalized": false,
                "single_word": false, "lstrip": false, "rstrip": false})
        })
        .collect();
    json!({
        "added_tokens": added_tokens,
        // SmolLM's split of text into words: each digit alone, then GPT-2's
        // rules, with no space added before the first word.
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Digits", "individual_digits": true},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
        ]},
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
        "model": {"type": "BPE", "vocab": ids, "merges": merges},
    })
}

/// The printable character that byte-level BPE gives each byte.
fn byte_characters() -> Vec<char> {
    let prints_as_itself = |byte| matches!(byte, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
    let mut next_character = 256;
    (0..=255)
        .map(|byte| {
            if prints_as_itself(byte) {
                char::from(byte)
            } else {
                next_character += 1;
                char::from_u32(next_character - 1).unwrap()
            }
        })
        .collect()
}
