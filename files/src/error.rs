//! What can go wrong reading or writing a model file.

use std::fmt;
use std::io;
use std::path::PathBuf;

/// An error from reading or writing a model file.
#[derive(Debug)]
pub enum Error {
    /// The file at `path` couldn't be read or written.
    Io {
        /// The file's path.
        path: PathBuf,
        /// Why not, as the operating system says.
        error: io::Error,
    },
    /// The file breaks its format's rules, or holds a tensor this package
    /// doesn't read, or a tensor can't be written. The message says what's
    /// wrong, and in which tensor.
    Invalid(String),
    /// Tensor `name` holds quantize's bytes, but
    /// [`Quantized::from_bytes`](quantize::Quantized::from_bytes) refused
    /// them, as when they were saved with another scale type.
    Quantized {
        /// The tensor's name.
        name: String,
        /// Why `from_bytes` refused them.
        error: quantize::Error,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, error } => write!(f, "{}: {error}", path.display()),
            Self::Invalid(message) => f.write_str(message),
            Self::Quantized { name, error } => write!(f, "tensor {name:?}: {error}"),
        }
    }
}

impl std::error::Error for Error {}

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}
