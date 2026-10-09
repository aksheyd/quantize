//! The tokenizer's metadata, which llama.cpp turns text into tokens and back
//! with, as llama.cpp's converter writes it for SmolLM-135M: the tokens and
//! merges from `tokenizer.json`, and the tokens that start and end a text
//! from `config.json`.

use quantize_files::gguf::Value;
use tokenizers::Tokenizer;

use crate::config::Config;

pub fn tokenizer_metadata(
    config: &Config,
    tokenizer: &Tokenizer,
) -> Result<Vec<(&'static str, Value)>, String> {
    // tokenizer.json's model, as the tokenizers crate writes it back.
    let model = serde_json::to_value(tokenizer.get_model()).map_err(|error| error.to_string())?;
    if model["type"] != "BPE" {
        let model_type = &model["type"];
        return Err(format!(
            "llama.cpp runs byte-level BPE tokenizers like SmolLM's, not {model_type}"
        ));
    }
    let (tokens, token_types) = tokens_and_types(config, tokenizer);
    let beginning = Value::U32(config.beginning_of_sequence_token);
    let end = Value::U32(config.end_of_sequence_token);
    Ok(vec![
        ("tokenizer.ggml.model", Value::String("gpt2".into())),
        // How SmolLM's tokenizer splits text into words for BPE: each digit
        // alone, then GPT-2's rules. llama.cpp has those rules built in, by
        // this name.
        ("tokenizer.ggml.pre", Value::String("smollm".into())),
        // SmolLM's tokenizer adds nothing to a text before splitting it: no
        // space before the first word, and no token before the first token.
        ("tokenizer.ggml.add_space_prefix", Value::Bool(false)),
        ("tokenizer.ggml.add_bos_token", Value::Bool(false)),
        ("tokenizer.ggml.tokens", tokens),
        ("tokenizer.ggml.token_type", token_types),
        ("tokenizer.ggml.merges", merges(&model["merges"])?),
        ("tokenizer.ggml.bos_token_id", beginning),
        ("tokenizer.ggml.eos_token_id", end),
    ])
}

/// llama.cpp's numbers for a token's type: an ordinary piece of text, a
/// special token that text never turns into, like `<|endoftext|>`, a token
/// added to the vocabulary that text does turn into, and a row of the
/// embedding table that no token uses.
const NORMAL: i32 = 1;
const CONTROL: i32 = 3;
const USER_DEFINED: i32 = 4;
const UNUSED: i32 = 5;

/// Each token's text, and its type, in order of id. The embedding table has
/// a row for every id below vocab_size, which the checkpoint checked every
/// token's id is, and the converter names an id without a token `[PAD<id>]`.
fn tokens_and_types(config: &Config, tokenizer: &Tokenizer) -> (Value, Value) {
    let mut texts: Vec<Option<String>> = vec![None; config.vocab_size];
    for (text, id) in tokenizer.get_vocab(true) {
        texts[id as usize] = Some(text);
    }
    let added = tokenizer.get_added_tokens_decoder();
    let (mut tokens, mut token_types) = (Vec::new(), Vec::new());
    for (id, text) in texts.into_iter().enumerate() {
        let (text, token_type) = match (text, added.get(&(id as u32))) {
            (None, _) => (format!("[PAD{id}]"), UNUSED),
            (Some(text), Some(token)) if token.special => (text, CONTROL),
            (Some(text), Some(_)) => (text, USER_DEFINED),
            (Some(text), None) => (text, NORMAL),
        };
        tokens.push(Value::String(text));
        token_types.push(Value::I32(token_type));
    }
    (Value::Array(tokens), Value::Array(token_types))
}

/// Each merge joins two tokens into one, in the order BPE tries them, as
/// `first second`. A byte-level token never holds a space: the space byte
/// has a character of its own, `Ġ`.
fn merges(pairs: &serde_json::Value) -> Result<Value, String> {
    let pairs = pairs.as_array().ok_or("the tokenizer has no merges")?;
    let merges = pairs
        .iter()
        .map(|pair| match (pair[0].as_str(), pair[1].as_str()) {
            (Some(first), Some(second)) => Ok(Value::String(format!("{first} {second}"))),
            _ => Err(format!("a merge should be a pair of tokens, not {pair}")),
        });
    Ok(Value::Array(merges.collect::<Result<_, _>>()?))
}
