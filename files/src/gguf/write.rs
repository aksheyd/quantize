//! Write a gguf file.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use half::f16;

use super::ggml_types::encode;
use super::{MAGIC, Metadata, VERSION, alignment};
use crate::Tensor;
use crate::error::{Error, invalid};

/// Write `metadata` and `tensors` to a gguf file at `path`, replacing any
/// file there.
///
/// A [`Tensor::Float`] is saved as `F32`, and a [`Tensor::Quantized`] as
/// `Q4_0` or `Q8_0`, with two dimensions. The tensors' data aligns to
/// `general.alignment`, if `metadata` has it, or to 32 bytes.
///
/// # Errors
///
/// [`Error::Invalid`] if `general.alignment` isn't a
/// [`Value::U32`](super::Value::U32) that is a power of two, an array is
/// empty or its elements don't all have one type, a float tensor's shape
/// doesn't hold its number of values, a quantized tensor isn't a symmetric
/// matrix with 4-bit or 8-bit codes in blocks of 32 that split its rows, or
/// the file would break one of the limits of ggml, which llama.cpp loads
/// files with: more than 4 dimensions, a tensor name of 64 bytes or more, or
/// an array of arrays. [`Error::Io`] if the file can't be written.
pub fn write(
    path: impl AsRef<Path>,
    metadata: &Metadata,
    tensors: &BTreeMap<String, Tensor<f16>>,
) -> Result<(), Error> {
    let path = path.as_ref();
    let bytes = to_bytes(metadata, tensors)?;
    fs::write(path, bytes).map_err(|error| Error::Io {
        path: path.to_owned(),
        error,
    })
}

/// The bytes of a whole gguf file that holds `metadata` and `tensors`.
pub(super) fn to_bytes(
    metadata: &Metadata,
    tensors: &BTreeMap<String, Tensor<f16>>,
) -> Result<Vec<u8>, Error> {
    let alignment = alignment(metadata)?;
    let mut bytes = MAGIC.to_vec();
    bytes.extend(VERSION.to_le_bytes());
    bytes.extend((tensors.len() as u64).to_le_bytes());
    bytes.extend((metadata.len() as u64).to_le_bytes());
    for (key, value) in metadata {
        write_string(&mut bytes, key);
        value.write(key, &mut bytes)?;
    }

    // Each tensor's info goes in the header, and its bytes in the data, each
    // tensor right after the one before it, padded to the alignment.
    let mut data = Vec::new();
    for (name, tensor) in tensors {
        let (ggml_type, shape, tensor_bytes) = encode(name, tensor)?;
        check_ggml_loads(name, &shape)?;
        write_string(&mut bytes, name);
        bytes.extend((shape.len() as u32).to_le_bytes());
        // Dimensions run innermost first, the reverse of the shape.
        for &length in shape.iter().rev() {
            bytes.extend((length as u64).to_le_bytes());
        }
        bytes.extend(ggml_type.to_le_bytes());
        bytes.extend((data.len() as u64).to_le_bytes());
        data.extend(tensor_bytes);
        data.resize(data.len().next_multiple_of(alignment), 0);
    }
    bytes.resize(bytes.len().next_multiple_of(alignment), 0);
    bytes.extend(data);
    Ok(bytes)
}

/// Write a string: its length in bytes, then its UTF-8 bytes.
pub(super) fn write_string(bytes: &mut Vec<u8>, text: &str) {
    bytes.extend((text.len() as u64).to_le_bytes());
    bytes.extend(text.as_bytes());
}

/// The most dimensions a tensor can have for ggml to load it.
const MOST_DIMENSIONS: usize = 4;

/// The longest tensor name, in bytes, that ggml loads: it keeps each name in
/// 64 bytes, the last of them a zero.
const LONGEST_NAME: usize = 63;

/// Check that ggml would load tensor `name` of `shape`. The format allows any
/// name and any number of dimensions, but ggml refuses a whole file for one
/// tensor past its limits.
fn check_ggml_loads(name: &str, shape: &[usize]) -> Result<(), Error> {
    if name.len() > LONGEST_NAME {
        return Err(invalid(format!(
            "tensor {name:?}'s name is {} bytes long, but ggml loads names of at most {LONGEST_NAME}",
            name.len()
        )));
    }
    if shape.len() > MOST_DIMENSIONS {
        return Err(invalid(format!(
            "tensor {name:?} has {} dimensions, but ggml loads at most {MOST_DIMENSIONS}",
            shape.len()
        )));
    }
    Ok(())
}
