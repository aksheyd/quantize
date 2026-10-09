mod refusals;
mod round_trips;

use std::path::PathBuf;

use quantize::Scale;

use crate::Tensor;

fn values(count: usize) -> Vec<f32> {
    (0..count).map(|i| (i as f32 * 0.37).sin()).collect()
}

fn float<S: Scale>(shape: &[usize]) -> Tensor<S> {
    let values = values(shape.iter().product());
    let shape = shape.to_vec();
    Tensor::Float { shape, values }
}

/// A whole file: the header's length, the header, then the data.
fn file(header: &str, data: &[u8]) -> Vec<u8> {
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header.as_bytes());
    bytes.extend(data);
    bytes
}

/// A path in the temporary directory that no other test run uses.
fn temporary_path(name: &str) -> PathBuf {
    let file_name = format!("quantize-files-{}-{name}.safetensors", std::process::id());
    std::env::temp_dir().join(file_name)
}
