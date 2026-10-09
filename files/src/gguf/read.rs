//! Read a gguf file, checking each of the format's rules on the way.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use half::f16;

use super::ggml_types::{byte_count_of, decode};
use super::reader::Reader;
use super::{MAGIC, Metadata, VERSION, Value, alignment};
use crate::Tensor;
use crate::error::{Error, invalid};

/// Read the metadata and every tensor in the gguf file at `path`, each by
/// name.
///
/// `F32`, `F16`, and `BF16` tensors read as [`Tensor::Float`], widened to
/// `f32`, with the shape outermost first. `Q4_0` and `Q8_0` tensors read as
/// [`Tensor::Quantized`]: symmetric matrices with 4-bit or 8-bit codes in
/// blocks of 32, whose rows run along the innermost dimension.
///
/// # Errors
///
/// [`Error::Io`] if the file can't be read, and [`Error::Invalid`] if it
/// isn't little-endian gguf version 3, breaks the format's rules, or holds a
/// tensor of another ggml type, or a `Q4_0` or `Q8_0` tensor whose rows
/// don't split into blocks of 32.
pub fn read(path: impl AsRef<Path>) -> Result<(Metadata, BTreeMap<String, Tensor<f16>>), Error> {
    let path = path.as_ref();
    let bytes = fs::read(path).map_err(|error| Error::Io {
        path: path.to_owned(),
        error,
    })?;
    from_bytes(&bytes)
}

/// Read the metadata and every tensor in the bytes of a whole gguf file.
pub(super) fn from_bytes(bytes: &[u8]) -> Result<(Metadata, BTreeMap<String, Tensor<f16>>), Error> {
    let mut reader = Reader { bytes, position: 0 };
    let (tensor_count, metadata_count) = read_header(&mut reader)?;
    let metadata = read_metadata(&mut reader, metadata_count)?;
    let alignment = alignment(&metadata)?;
    let tensor_infos = read_tensor_infos(&mut reader, tensor_count)?;
    let data_start = reader.position.next_multiple_of(alignment);
    let tensors = read_data(bytes, data_start, alignment, tensor_infos)?;
    Ok((metadata, tensors))
}

/// Check the magic and the version, and read how many tensors and metadata
/// entries the file holds.
fn read_header(reader: &mut Reader) -> Result<(u64, u64), Error> {
    if reader.take(MAGIC.len())? != MAGIC {
        return Err(invalid("the file doesn't start with GGUF"));
    }
    let version = reader.bytes()?;
    if u32::from_be_bytes(version) == VERSION {
        return Err(invalid(
            "the file is big-endian gguf, but only little-endian is read",
        ));
    }
    let version = u32::from_le_bytes(version);
    if version != VERSION {
        return Err(invalid(format!(
            "the file is gguf version {version}, but only version {VERSION} is read"
        )));
    }
    Ok((reader.u64()?, reader.u64()?))
}

fn read_metadata(reader: &mut Reader, count: u64) -> Result<Metadata, Error> {
    let mut metadata = BTreeMap::new();
    for _ in 0..count {
        let key = reader.string()?;
        let value = Value::read(reader, &key)?;
        if metadata.insert(key.clone(), value).is_some() {
            return Err(invalid(format!("metadata {key:?} appears twice")));
        }
    }
    Ok(metadata)
}

/// What the file says about a tensor before the data.
struct TensorInfo {
    name: String,
    /// Outermost first, the reverse of how the file lists the dimensions.
    shape: Vec<usize>,
    ggml_type: u32,
    /// Where its data starts, counting from the start of the data.
    offset: usize,
}

fn read_tensor_infos(reader: &mut Reader, count: u64) -> Result<Vec<TensorInfo>, Error> {
    let mut tensor_infos = Vec::new();
    for _ in 0..count {
        let name = reader.string()?;
        let mut shape = Vec::new();
        for _ in 0..reader.u32()? {
            shape.push(reader.size()?);
        }
        shape.reverse();
        let ggml_type = reader.u32()?;
        let offset = reader.size()?;
        tensor_infos.push(TensorInfo {
            name,
            shape,
            ggml_type,
            offset,
        });
    }
    Ok(tensor_infos)
}

/// Read each tensor's data, which starts at byte `data_start` of the file
/// with the first tensor's, each tensor right after the one before it,
/// padded to `alignment`.
fn read_data(
    bytes: &[u8],
    data_start: usize,
    alignment: usize,
    tensor_infos: Vec<TensorInfo>,
) -> Result<BTreeMap<String, Tensor<f16>>, Error> {
    let mut tensors = BTreeMap::new();
    let mut next_offset = 0;
    for info in tensor_infos {
        let name = info.name;
        let byte_count = byte_count_of(&name, info.ggml_type, &info.shape)?;
        let offset = info.offset;
        if !offset.is_multiple_of(alignment) {
            return Err(invalid(format!(
                "tensor {name:?} starts at byte {offset} of the data, which isn't a multiple of the alignment, {alignment}"
            )));
        }
        if offset != next_offset {
            return Err(invalid(format!(
                "tensor {name:?} starts at byte {offset} of the data, but the tensors before it, padded to the alignment, end at byte {next_offset}"
            )));
        }
        let start = data_start + offset;
        let end = start.checked_add(byte_count);
        let Some(tensor_bytes) = end.and_then(|end| bytes.get(start..end)) else {
            return Err(invalid(format!(
                "tensor {name:?}'s data runs past the end of the file, at byte {}",
                bytes.len()
            )));
        };
        next_offset = (offset + byte_count).next_multiple_of(alignment);
        let tensor = decode(info.ggml_type, info.shape, tensor_bytes);
        if tensors.insert(name.clone(), tensor).is_some() {
            return Err(invalid(format!("tensor {name:?} appears twice")));
        }
    }
    Ok(tensors)
}
