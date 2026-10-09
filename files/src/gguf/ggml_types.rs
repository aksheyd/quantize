//! The ggml types this package reads and writes, and how a tensor's bytes
//! turn into a [`Tensor`] and back.

use half::{bf16, f16};

use super::blocks::BLOCK;
use super::quantized::{GgmlType, blocks_of, check_rows, ggml_type_of, tensor_of};
use super::value_count;
use crate::Tensor;
use crate::error::{Error, invalid};

const F32: u32 = 0;
const F16: u32 = 1;
const Q4_0: u32 = GgmlType::Q4_0 as u32;
const Q8_0: u32 = GgmlType::Q8_0 as u32;
const BF16: u32 = 30;

/// How many bytes tensor `name`'s data takes: its shape's values, of
/// `ggml_type`.
pub(super) fn byte_count_of(name: &str, ggml_type: u32, shape: &[usize]) -> Result<usize, Error> {
    let count = value_count(shape);
    let byte_count = match ggml_type {
        F32 => count.and_then(|count| count.checked_mul(4)),
        F16 | BF16 => count.and_then(|count| count.checked_mul(2)),
        Q4_0 | Q8_0 => {
            let quantized_type = quantized_type(ggml_type);
            let subject = format!("tensor {name:?}");
            check_rows(&subject, quantized_type, row_length(shape))?;
            count.and_then(|count| (count / BLOCK).checked_mul(quantized_type.block_bytes()))
        }
        _ => {
            return Err(invalid(format!(
                "tensor {name:?} has ggml type {ggml_type}, but only F32 (0), F16 (1), Q4_0 (2), Q8_0 (8), and BF16 (30) are read"
            )));
        }
    };
    byte_count.ok_or_else(|| {
        invalid(format!(
            "tensor {name:?} has shape {shape:?}, which holds too many bytes to count"
        ))
    })
}

/// A `Q4_0` or `Q8_0` tensor is a matrix whose rows run along its innermost
/// dimension, the last of its shape, and its blocks run along the rows, so
/// each row must split into blocks of 32. Its outer dimensions, if it has
/// more than one, run together into rows, as ggml lays them out.
fn row_length(shape: &[usize]) -> usize {
    shape.last().copied().unwrap_or(1)
}

/// `Q4_0` or `Q8_0`, as the [`GgmlType`] it is.
fn quantized_type(ggml_type: u32) -> GgmlType {
    if ggml_type == Q4_0 {
        GgmlType::Q4_0
    } else {
        GgmlType::Q8_0
    }
}

/// Turn `bytes`, which hold `shape`'s values of `ggml_type`, into a
/// [`Tensor`].
pub(super) fn decode(ggml_type: u32, shape: Vec<usize>, bytes: &[u8]) -> Tensor<f16> {
    let values = match ggml_type {
        F32 => values_of(bytes, f32::from_le_bytes),
        F16 => values_of(bytes, |value| f16::from_le_bytes(value).to_f32()),
        BF16 => values_of(bytes, |value| bf16::from_le_bytes(value).to_f32()),
        // Only Q4_0 and Q8_0 are left, since `byte_count_of` refused every
        // other type, and checked their rows.
        _ => {
            let matrix = tensor_of(bytes, quantized_type(ggml_type), row_length(&shape));
            return Tensor::Quantized(matrix);
        }
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
        Tensor::Quantized(quantized) => {
            let subject = format!("tensor {name:?}");
            let (quantized_type, rows, columns) = ggml_type_of(&subject, quantized)?;
            let bytes = blocks_of(quantized, quantized_type);
            Ok((quantized_type as u32, vec![rows, columns], bytes))
        }
    }
}

/// Every `N`-byte value in `bytes`, turned into an `f32` by `to_f32`. The
/// values fill `bytes` exactly, as the reader checked.
fn values_of<const N: usize>(bytes: &[u8], to_f32: fn([u8; N]) -> f32) -> Vec<f32> {
    let (values, _) = bytes.as_chunks::<N>();
    values.iter().map(|&value| to_f32(value)).collect()
}
