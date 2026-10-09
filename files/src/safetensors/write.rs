//! Write a safetensors file.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use quantize::Scale;
use serde_json::{Map, Value, json};

use super::value_count;
use crate::Tensor;
use crate::error::{Error, invalid};

/// Write `tensors` to a safetensors file at `path`, replacing any file there.
///
/// A [`Tensor::Float`] is saved as `F32`, and a [`Tensor::Quantized`] as a
/// one-dimensional `U8` tensor holding the bytes of
/// [`Quantized::to_bytes`](quantize::Quantized::to_bytes).
///
/// # Errors
///
/// [`Error::Invalid`] if a float tensor's shape doesn't hold its number of
/// values, or a tensor is named `__metadata__`, which the format keeps for
/// metadata, and [`Error::Io`] if the file can't be written.
pub fn write<S: Scale>(
    path: impl AsRef<Path>,
    tensors: &BTreeMap<String, Tensor<S>>,
) -> Result<(), Error> {
    let path = path.as_ref();
    let bytes = to_bytes(tensors)?;
    fs::write(path, bytes).map_err(|error| Error::Io {
        path: path.to_owned(),
        error,
    })
}

/// The bytes of a whole safetensors file that holds `tensors`.
pub(super) fn to_bytes<S: Scale>(tensors: &BTreeMap<String, Tensor<S>>) -> Result<Vec<u8>, Error> {
    // The floats go first. The data starts on a multiple of 8 bytes, so
    // each F32 tensor then starts on a multiple of 4, and a program that maps
    // the file into memory can use its values where they are.
    let mut floats_first: Vec<_> = tensors.iter().collect();
    floats_first.sort_by_key(|(_, tensor)| matches!(tensor, Tensor::Quantized(_)));

    let mut header = Map::new();
    let mut data = Vec::new();
    for (name, tensor) in floats_first {
        if name == "__metadata__" {
            return Err(invalid("a tensor can't be named __metadata__"));
        }
        let begin = data.len();
        let (dtype, shape) = match tensor {
            Tensor::Float { shape, values } => {
                if value_count(shape) != Some(values.len()) {
                    return Err(invalid(format!(
                        "tensor {name:?} has {} values, which don't fit its shape {shape:?}",
                        values.len()
                    )));
                }
                data.extend(values.iter().flat_map(|value| value.to_le_bytes()));
                ("F32", shape.clone())
            }
            Tensor::Quantized(quantized) => {
                data.extend(quantized.to_bytes());
                ("U8", vec![data.len() - begin])
            }
        };
        let entry = json!({"dtype": dtype, "shape": shape, "data_offsets": [begin, data.len()]});
        header.insert(name.clone(), entry);
    }

    // Spaces pad the header to a multiple of 8 bytes, as the safetensors
    // package pads it, so the data after it starts on a multiple of 8 too.
    let mut header = Value::Object(header).to_string().into_bytes();
    header.resize(header.len().next_multiple_of(8), b' ');
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(data);
    Ok(bytes)
}
