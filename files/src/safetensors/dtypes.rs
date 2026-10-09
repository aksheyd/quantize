//! The dtypes this package reads, and how their bytes turn into a [`Tensor`].

use half::{bf16, f16};
use quantize::{Quantized, Scale};

use crate::Tensor;
use crate::error::{Error, invalid};

/// How many bytes each value of `dtype` takes, or `None` for a dtype that
/// this package doesn't read.
pub(super) fn value_size(dtype: &str) -> Option<usize> {
    match dtype {
        "F32" => Some(4),
        "F16" | "BF16" => Some(2),
        "U8" => Some(1),
        _ => None,
    }
}

/// Turn the bytes of tensor `name`, which hold its shape's values of `dtype`,
/// into a [`Tensor`]: floats widened to `f32`, or the quantized tensor that a
/// `U8` tensor's bytes hold.
pub(super) fn decode<S: Scale>(
    name: &str,
    dtype: &str,
    shape: Vec<usize>,
    bytes: &[u8],
) -> Result<Tensor<S>, Error> {
    let values = match dtype {
        "F32" => values_of(bytes, f32::from_le_bytes),
        "F16" => values_of(bytes, |value| f16::from_le_bytes(value).to_f32()),
        "BF16" => values_of(bytes, |value| bf16::from_le_bytes(value).to_f32()),
        // Only U8 is left, since `value_size` knows no other dtype.
        _ if bytes.starts_with(b"QNTZ") => {
            return Quantized::from_bytes(bytes)
                .map(Tensor::Quantized)
                .map_err(|error| Error::Quantized {
                    name: name.to_string(),
                    error,
                });
        }
        _ => {
            return Err(invalid(format!(
                "tensor {name:?} is U8, but doesn't start with QNTZ, so it isn't a tensor that quantize saved"
            )));
        }
    };
    Ok(Tensor::Float { shape, values })
}

/// Every `N`-byte value in `bytes`, turned into an `f32` by `to_f32`. The
/// values fill `bytes` exactly, as the reader checked.
fn values_of<const N: usize>(bytes: &[u8], to_f32: fn([u8; N]) -> f32) -> Vec<f32> {
    let (values, _) = bytes.as_chunks::<N>();
    values.iter().map(|&value| to_f32(value)).collect()
}
