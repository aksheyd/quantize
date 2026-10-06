//! Feature-gated WikiText runner. Not part of the published crate.

pub mod data;
pub mod eval;
pub mod net;

pub const MODEL_ID: &str = "HuggingFaceTB/SmolLM-135M";
pub const CONTEXT: usize = 512;
pub const STRIDE: usize = 256;
pub const DECODE_PROMPT: usize = 128;
pub const DECODE_NEW: usize = 128;

pub fn candle_msg(error: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::msg(error)
}

/// ureq's errors don't say which URL failed, so download errors start with it.
pub fn error_with_url(url: &str, error: impl std::fmt::Display) -> candle_core::Error {
    candle_msg(format!("{url}: {error}"))
}
