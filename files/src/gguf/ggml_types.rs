//! The ggml types this package reads and writes, and how a tensor's bytes
//! turn into a [`Tensor`] and back.

use half::{bf16, f16};

use super::value_count;
use crate::Tensor;
use crate::error::{Error, invalid};

const F32: u32 = 0;
const F16: u32 = 1;
const BF16: u32 = 30;

/// How many bytes tensor `name`'s data takes: its shape's values, of
/// `ggml_type`.
pub(super) fn data_size(name: &str, ggml_type: u32, shape: &[usize]) -> Result<usize, Error> {
    let value_size = match ggml_type {
        F32 => 4,
        F16 | BF16 => 2,
        _ => {
            return Err(invalid(format!(
                "tensor {name:?} has ggml type {ggml_type}, but only F32 (0), F16 (1), and BF16 (30) are read"
            )));
        }
    };
    let size = value_count(shape).and_then(|count| count.checked_mul(value_size));
    size.ok_or_else(|| {
        invalid(format!(
            "tensor {name:?} has shape {shape:?}, which holds too many bytes to count"
        ))
    })
}

/// Turn `bytes`, which hold `shape`'s values of `ggml_type`, into a
/// [`Tensor`].
pub(super) fn decode(ggml_type: u32, shape: Vec<usize>, bytes: &[u8]) -> Tensor<f16> {
    let values = match ggml_type {
        F32 => values_of(bytes, f32::from_le_bytes),
        F16 => values_of(bytes, |value| f16::from_le_bytes(value).to_f32()),
        // Only BF16 is left, since `data_size` refused every other type.
        _ => values_of(bytes, |value| bf16::from_le_bytes(value).to_f32()),
    };
    Tensor::Float { shape, values }
}

/// The ggml type, the shape, and the bytes that tensor `name` is written as.
pub(super) fn encode(
    name: &str,
    tensor: &Tensor<f16>,
) -> Result<(u32, Vec<usize>, Vec<u8>), Error> {
    match tensor {
        Tensor::Float { shape, values } => {
            if value_count(shape) != Some(values.len()) {
                return Err(invalid(format!(
                    "tensor {name:?} has {} values, which don't fit its shape {shape:?}",
                    values.len()
                )));
            }
            let bytes = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            Ok((F32, shape.clone(), bytes))
        }
        Tensor::Quantized(_) => Err(invalid(format!(
            "tensor {name:?} is quantized, but only float tensors are written to gguf"
        ))),
    }
}

/// Every `N`-byte value in `bytes`, turned into an `f32` by `to_f32`. The
/// values fill `bytes` exactly, as the reader checked.
fn values_of<const N: usize>(bytes: &[u8], to_f32: fn([u8; N]) -> f32) -> Vec<f32> {
    let (values, _) = bytes.as_chunks::<N>();
    values.iter().map(|&value| to_f32(value)).collect()
}
