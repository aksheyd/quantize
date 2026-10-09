//! The command line: the prompt, and two optional flags.

use std::path::PathBuf;

use quantize::Scheme;

const USAGE: &str = "usage: smollm [--model DIR] [--scheme SCHEME] PROMPT";

pub struct Arguments {
    /// The text to continue.
    pub prompt: String,
    /// `--model DIR`: read `config.json`, `tokenizer.json`, and
    /// `model.safetensors` from `DIR` instead of downloading them.
    pub model_directory: Option<PathBuf>,
    /// `--scheme SCHEME`: how to quantize the weights, as
    /// `str::parse::<Scheme>` reads it: `Q4_32` (the default), `Q8_32`, or
    /// text like `symmetric(bits=4)`.
    pub scheme: Scheme,
}

impl Arguments {
    /// Read the arguments after the program's name. Every argument that
    /// isn't a flag is a word of the prompt, so it needs no quotes.
    pub fn parse(mut arguments: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut words = Vec::new();
        let mut model_directory = None;
        let mut scheme = Scheme::Q4_32;
        while let Some(argument) = arguments.next() {
            let mut value = || {
                arguments
                    .next()
                    .ok_or(format!("{argument} needs a value; {USAGE}"))
            };
            match argument.as_str() {
                "--model" => model_directory = Some(PathBuf::from(value()?)),
                "--scheme" => {
                    scheme = value()?
                        .parse()
                        .map_err(|error| format!("--scheme: {error}"))?
                }
                flag if flag.starts_with("--") => {
                    return Err(format!("unknown flag {flag}; {USAGE}"));
                }
                _ => words.push(argument),
            }
        }
        if words.is_empty() {
            return Err(USAGE.into());
        }
        Ok(Self {
            prompt: words.join(" "),
            model_directory,
            scheme,
        })
    }
}
