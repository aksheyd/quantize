mod quantized;
mod refusals;
mod round_trips;

use std::path::PathBuf;

use half::f16;

use crate::Tensor;

fn values(count: usize) -> Vec<f32> {
    (0..count).map(|i| (i as f32 * 0.37).sin()).collect()
}

fn float(shape: &[usize]) -> Tensor<f16> {
    let values = values(shape.iter().product());
    let shape = shape.to_vec();
    Tensor::Float { shape, values }
}

/// A string: its length in 8 bytes, then its UTF-8 bytes.
fn string(text: &str) -> Vec<u8> {
    [&(text.len() as u64).to_le_bytes(), text.as_bytes()].concat()
}

/// A metadata entry: its key, its value's type, then the value's bytes.
fn entry(key: &str, value_type: u32, value: &[u8]) -> Vec<u8> {
    [&string(key), &value_type.to_le_bytes()[..], value].concat()
}

/// A tensor's info: its name, its dimensions, innermost first, its ggml type,
/// and its offset.
fn tensor_info(name: &str, dimensions: &[u64], ggml_type: u32, offset: u64) -> Vec<u8> {
    let mut bytes = string(name);
    bytes.extend((dimensions.len() as u32).to_le_bytes());
    for dimension in dimensions {
        bytes.extend(dimension.to_le_bytes());
    }
    bytes.extend(ggml_type.to_le_bytes());
    bytes.extend(offset.to_le_bytes());
    bytes
}

/// A whole file: the header, the metadata entries, the tensor infos, zeros up
/// to a multiple of 32 bytes, then `data`.
fn file(entries: &[Vec<u8>], tensor_infos: &[Vec<u8>], data: &[u8]) -> Vec<u8> {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3_u32.to_le_bytes());
    bytes.extend((tensor_infos.len() as u64).to_le_bytes());
    bytes.extend((entries.len() as u64).to_le_bytes());
    bytes.extend(entries.concat());
    bytes.extend(tensor_infos.concat());
    bytes.resize(bytes.len().next_multiple_of(32), 0);
    bytes.extend(data);
    bytes
}

/// The bytes of `values` as `F32`.
fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// A path in the temporary directory that no other test run uses.
fn temporary_path(name: &str) -> PathBuf {
    let file_name = format!("quantize-files-{}-{name}.gguf", std::process::id());
    std::env::temp_dir().join(file_name)
}
