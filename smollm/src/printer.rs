//! Prints generated text as its tokens arrive.

use std::io::{self, Write};

use tokenizers::Tokenizer;

/// A token can end partway through a character, which decodes as "�" until
/// the next token finishes it. So each time a token arrives, this decodes
/// every token so far and prints what's new, once the text ends in a whole
/// character.
pub struct TextPrinter<'a> {
    tokenizer: &'a Tokenizer,
    tokens: Vec<u32>,
    printed: usize,
}

impl<'a> TextPrinter<'a> {
    /// Start with the prompt's tokens, and print their text.
    pub fn new(tokenizer: &'a Tokenizer, prompt: &[u32]) -> Self {
        let mut printer = Self {
            tokenizer,
            tokens: prompt.to_vec(),
            printed: 0,
        };
        printer.print_new_text();
        printer
    }

    pub fn push(&mut self, token: u32) {
        self.tokens.push(token);
        self.print_new_text();
    }

    fn print_new_text(&mut self) {
        let Ok(text) = self.tokenizer.decode(&self.tokens, true) else {
            return;
        };
        if text.ends_with('\u{FFFD}') {
            return;
        }
        print!("{}", text.get(self.printed..).unwrap_or_default());
        self.printed = text.len();
        io::stdout().flush().ok();
    }
}
