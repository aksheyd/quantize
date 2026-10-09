//! quantize's tensors as ggml's `Q4_0` and `Q8_0` blocks, outside a file.
//!
//! [`to_ggml`] gives the bytes that a gguf file holds for a tensor, and
//! [`from_ggml`] turns them back into one, for a file that something else
//! reads or writes, like the `gguf` Python package. [`write()`](super::write())
//! and [`read()`](super::read()) check each tensor here too, naming it in
//! their errors.

use std::str::FromStr;

use half::f16;
use quantize::Quantized;

use super::blocks::{
    BLOCK, Q4_0_BLOCK_BYTES, Q8_0_BLOCK_BYTES, from_q4_0, from_q8_0, q4_0_blocks, q8_0_blocks,
};
use crate::error::{Error, invalid};

/// A ggml type that holds quantize's tensors, numbered as gguf files store
/// it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    /// [`Scheme::Q4_32`](quantize::Scheme::Q4_32) with f16 scales.
    Q4_0 = 2,
    /// [`Scheme::Q8_32`](quantize::Scheme::Q8_32) with f16 scales.
    Q8_0 = 8,
}

impl GgmlType {
    /// Its name, as ggml and the `gguf` Python package write it: `"Q4_0"` or
    /// `"Q8_0"`, which [`parse`](str::parse) reads back.
    pub fn name(self) -> &'static str {
        match self {
            Self::Q4_0 => "Q4_0",
            Self::Q8_0 => "Q8_0",
        }
    }

    /// How many bytes each block of [`BLOCK`] values takes, its scale and
    /// then its codes: 18 for `Q4_0`, and 34 for `Q8_0`.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::Q4_0 => Q4_0_BLOCK_BYTES,
            Self::Q8_0 => Q8_0_BLOCK_BYTES,
        }
    }
}

impl FromStr for GgmlType {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self, Error> {
        match name {
            "Q4_0" => Ok(Self::Q4_0),
            "Q8_0" => Ok(Self::Q8_0),
            _ => Err(invalid(format!(
                "{name:?} isn't Q4_0 or Q8_0, the ggml types that hold quantize's tensors"
            ))),
        }
    }
}

/// The blocks that hold `tensor`, row after row, as a gguf file holds them,
/// and their type: `Q4_0` for 4-bit codes, and `Q8_0` for 8-bit ones. Each
/// row of `columns` values takes `columns / 32` blocks.
///
/// # Errors
///
/// [`Error::Invalid`] if `tensor` isn't a symmetric matrix with 4-bit or
/// 8-bit codes in blocks of 32 that split its rows, saying why.
pub fn to_ggml(tensor: &Quantized<f16>) -> Result<(Vec<u8>, GgmlType), Error> {
    let (ggml_type, _, _) = ggml_type_of("the tensor", tensor)?;
    Ok((blocks_of(tensor, ggml_type), ggml_type))
}

/// The matrix of `rows` rows of `columns` values that `blocks` of
/// `ggml_type` hold, row after row, as [`to_ggml`] gives them: symmetric,
/// with 4-bit or 8-bit codes in blocks of 32, and its shape set.
///
/// # Errors
///
/// [`Error::Invalid`] if `columns` is 0 or doesn't split into blocks of 32,
/// or `blocks` isn't `rows` rows of them.
pub fn from_ggml(
    blocks: &[u8],
    ggml_type: GgmlType,
    rows: usize,
    columns: usize,
) -> Result<Quantized<f16>, Error> {
    check_rows("the tensor", ggml_type, columns)?;
    let row_bytes = columns / BLOCK * ggml_type.block_bytes();
    if rows.checked_mul(row_bytes) != Some(blocks.len()) {
        return Err(invalid(format!(
            "{} bytes aren't {rows} rows of {columns} {} values, which take {row_bytes} bytes each",
            blocks.len(),
            ggml_type.name()
        )));
    }
    Ok(tensor_of(blocks, ggml_type, columns))
}

/// Which ggml type holds quantized tensor `subject`, like `the tensor` or
/// `tensor "w"`, with its rows and columns, or why neither can.
pub(super) fn ggml_type_of(
    subject: &str,
    tensor: &Quantized<f16>,
) -> Result<(GgmlType, usize, usize), Error> {
    let cannot_hold = |why: &str| {
        invalid(format!(
            "{subject} {why}; safetensors keeps any quantized tensor"
        ))
    };
    let bits = match tensor {
        Quantized::Symmetric { codes, .. } => codes.bits(),
        Quantized::Asymmetric { .. } => {
            return Err(cannot_hold(
                "is asymmetric, but Q4_0 and Q8_0 are symmetric",
            ));
        }
        Quantized::Adaptive { .. } => {
            return Err(cannot_hold(
                "is adaptive, but Q4_0 and Q8_0 give every block one width",
            ));
        }
    };
    let ggml_type = match bits {
        4 => GgmlType::Q4_0,
        8 => GgmlType::Q8_0,
        _ => {
            let why = format!("has {bits}-bit codes, but Q4_0 and Q8_0 have 4-bit and 8-bit ones");
            return Err(cannot_hold(&why));
        }
    };
    let block = tensor.block();
    if block != BLOCK {
        let why = format!("has blocks of {block}, but Q4_0 and Q8_0 have blocks of {BLOCK}");
        return Err(cannot_hold(&why));
    }
    let Some((rows, columns)) = tensor.shape() else {
        let why = format!(
            "is a flat vector of {} values, but Q4_0 and Q8_0 blocks run along a matrix's rows",
            tensor.len()
        );
        return Err(cannot_hold(&why));
    };
    if !columns.is_multiple_of(BLOCK) {
        let why = format!(
            "has rows of {columns} values, but Q4_0 and Q8_0 split each row into blocks of {BLOCK}"
        );
        return Err(cannot_hold(&why));
    }
    Ok((ggml_type, rows, columns))
}

/// Check that rows of `columns` values split into `ggml_type` blocks, as
/// they must for `subject`, like `the tensor` or `tensor "w"`.
pub(super) fn check_rows(subject: &str, ggml_type: GgmlType, columns: usize) -> Result<(), Error> {
    let type_name = ggml_type.name();
    match columns {
        0 => Err(invalid(format!(
            "{subject} is {type_name} with rows of no values, which quantize's tensors can't hold"
        ))),
        _ if !columns.is_multiple_of(BLOCK) => Err(invalid(format!(
            "{subject} is {type_name} with rows of {columns} values, which don't split into blocks of {BLOCK}"
        ))),
        _ => Ok(()),
    }
}

/// The blocks of `tensor`, which [`ggml_type_of`] found `ggml_type` holds.
pub(super) fn blocks_of(tensor: &Quantized<f16>, ggml_type: GgmlType) -> Vec<u8> {
    match ggml_type {
        GgmlType::Q4_0 => q4_0_blocks(tensor),
        GgmlType::Q8_0 => q8_0_blocks(tensor),
    }
}

/// The matrix, with rows of `columns` values, that `ggml_type` `blocks`
/// hold, once [`check_rows`] has passed and `blocks` holds whole rows.
pub(super) fn tensor_of(blocks: &[u8], ggml_type: GgmlType, columns: usize) -> Quantized<f16> {
    match ggml_type {
        GgmlType::Q4_0 => from_q4_0(blocks, columns),
        GgmlType::Q8_0 => from_q8_0(blocks, columns),
    }
}
