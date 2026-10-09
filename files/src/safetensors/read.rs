//! Read a safetensors file, checking each of the format's rules on the way.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use quantize::Scale;
use serde_json::{Map, Value};

use super::dtypes::{decode, value_size};
use super::value_count;
use crate::Tensor;
use crate::error::{Error, invalid};

/// Read every tensor in the safetensors file at `path`, by name.
///
/// `F32`, `F16`, and `BF16` tensors read as [`Tensor::Float`], widened to
/// `f32`. A `U8` tensor that starts with `QNTZ` reads as a
/// [`Tensor::Quantized`], through
/// [`Quantized::from_bytes`](quantize::Quantized::from_bytes), so `S` must be
/// the scale type it was saved with.
///
/// # Errors
///
/// [`Error::Io`] if the file can't be read, [`Error::Invalid`] if it breaks
/// the format's rules or holds a tensor of another dtype, or a `U8` tensor
/// that doesn't start with `QNTZ`, and [`Error::Quantized`] if a quantized
/// tensor doesn't load with scale type `S`.
pub fn read<S: Scale>(path: impl AsRef<Path>) -> Result<BTreeMap<String, Tensor<S>>, Error> {
    let path = path.as_ref();
    let bytes = fs::read(path).map_err(|error| Error::Io {
        path: path.to_owned(),
        error,
    })?;
    from_bytes(&bytes)
}

/// Read every tensor in the bytes of a whole safetensors file.
pub(super) fn from_bytes<S: Scale>(bytes: &[u8]) -> Result<BTreeMap<String, Tensor<S>>, Error> {
    let (header, data) = split_header(bytes)?;
    let mut tensors = BTreeMap::new();
    let mut byte_ranges = Vec::new();
    for (name, entry) in &header {
        if name == "__metadata__" {
            let strings = entry
                .as_object()
                .map(|metadata| metadata.values().all(Value::is_string));
            if strings != Some(true) {
                return Err(invalid("__metadata__ must map strings to strings"));
            }
            continue;
        }
        let dtype = entry["dtype"].as_str();
        let shape = sizes(&entry["shape"]);
        let offsets = sizes(&entry["data_offsets"]);
        let (Some(dtype), Some(shape), Some(&[begin, end])) = (dtype, shape, offsets.as_deref())
        else {
            return Err(invalid(format!(
                "tensor {name:?} needs a dtype, a shape, and two data_offsets"
            )));
        };
        let Some(value_size) = value_size(dtype) else {
            return Err(invalid(format!(
                "tensor {name:?} is {dtype}, but only F32, F16, BF16, and quantize's U8 tensors are read"
            )));
        };
        let size = value_count(&shape).and_then(|count| count.checked_mul(value_size));
        if begin > end || size != Some(end - begin) {
            return Err(invalid(format!(
                "tensor {name:?}'s data_offsets [{begin}, {end}] don't hold {dtype} values of shape {shape:?}"
            )));
        }
        let Some(tensor_bytes) = data.get(begin..end) else {
            return Err(invalid(format!(
                "tensor {name:?}'s data_offsets [{begin}, {end}] run past the end of the data, at byte {}",
                data.len()
            )));
        };
        byte_ranges.push((begin, end, name));
        tensors.insert(name.clone(), decode(name, dtype, shape, tensor_bytes)?);
    }
    check_no_overlaps_or_holes(byte_ranges, data.len())?;
    Ok(tensors)
}

/// The header's entries, by name, and the data after them.
fn split_header(bytes: &[u8]) -> Result<(Map<String, Value>, &[u8]), Error> {
    let Some((header_length, rest)) = bytes.split_first_chunk() else {
        return Err(invalid("the file is too short to hold its header's length"));
    };
    let header_length = u64::from_le_bytes(*header_length);
    let Some((header, data)) = usize::try_from(header_length)
        .ok()
        .and_then(|length| rest.split_at_checked(length))
    else {
        return Err(invalid(format!(
            "the header should be {header_length} bytes long, but the file has {} after its length",
            rest.len()
        )));
    };
    if !header.starts_with(b"{") {
        return Err(invalid("the header doesn't start with {"));
    }
    let header = serde_json::from_slice(header)
        .map_err(|error| invalid(format!("the header isn't a JSON object: {error}")))?;
    Ok((header, data))
}

/// A JSON array of sizes, like a shape, or `None` if it's anything else.
fn sizes(value: &Value) -> Option<Vec<usize>> {
    let array = value.as_array()?;
    array
        .iter()
        .map(|size| usize::try_from(size.as_u64()?).ok())
        .collect()
}

/// Check that the tensors' byte ranges cover the data end to end, as the
/// format requires: no byte belongs to two tensors, or to none. serde_json
/// keeps only the last entry for a name that the header gives twice, so
/// unless both entries hold the same bytes, the earlier one's show up here as
/// bytes that no tensor holds.
fn check_no_overlaps_or_holes(
    mut byte_ranges: Vec<(usize, usize, &String)>,
    data_length: usize,
) -> Result<(), Error> {
    byte_ranges.sort();
    let mut covered = 0;
    for (begin, end, name) in byte_ranges {
        if begin < covered {
            return Err(invalid(format!(
                "tensor {name:?} starts at byte {begin} of the data, inside the tensor before it"
            )));
        }
        if begin > covered {
            return Err(unclaimed(covered, begin));
        }
        covered = end;
    }
    if covered < data_length {
        return Err(unclaimed(covered, data_length));
    }
    Ok(())
}

fn unclaimed(begin: usize, end: usize) -> Error {
    let count = end - begin;
    invalid(format!(
        "no tensor holds the {count} bytes of the data from byte {begin}"
    ))
}
