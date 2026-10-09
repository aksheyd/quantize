//! Turn a tensor into ggml's `Q4_0` or `Q8_0` blocks and back, through
//! quantize-files, in the shape that the `gguf` package's `GGUFWriter`
//! takes and its `GGUFReader` gives: `(rows, bytes per row)`.

use std::sync::Arc;

use pyo3::prelude::*;
use quantize_files::gguf::{self, GgmlType};

use super::inner::QuantizedInner;
use crate::error::{QuantizeError, from_files};

/// The blocks that hold `inner`, with their shape, `(rows, bytes per row)`,
/// and the name of their ggml type.
pub(crate) fn to_ggml(inner: &QuantizedInner) -> PyResult<(Vec<u8>, [usize; 2], &'static str)> {
    let QuantizedInner::F16(tensor) = inner else {
        return Err(PyErr::new::<QuantizeError, _>(format!(
            "the tensor has {} scales, but Q4_0 and Q8_0 hold f16 ones; safetensors keeps any quantized tensor",
            inner.scale().name()
        )));
    };
    let (blocks, ggml_type) = gguf::to_ggml(tensor).map_err(from_files)?;
    let (rows, columns) = tensor
        .shape()
        .expect("to_ggml refuses a tensor without a shape");
    let row_bytes = columns / gguf::BLOCK * ggml_type.block_bytes();
    Ok((blocks, [rows, row_bytes], ggml_type.name()))
}

/// The tensor that `rows` rows of `row_bytes` bytes each hold, in blocks of
/// the ggml type named `ggml_type`.
pub(crate) fn from_ggml(
    blocks: &[u8],
    rows: usize,
    row_bytes: usize,
    ggml_type: &str,
) -> PyResult<QuantizedInner> {
    let ggml_type: GgmlType = ggml_type.parse().map_err(from_files)?;
    let block_bytes = ggml_type.block_bytes();
    if !row_bytes.is_multiple_of(block_bytes) {
        return Err(PyErr::new::<QuantizeError, _>(format!(
            "data has rows of {row_bytes} bytes, which don't split into {} blocks of {block_bytes} bytes",
            ggml_type.name()
        )));
    }
    let columns = row_bytes / block_bytes * gguf::BLOCK;
    let tensor = gguf::from_ggml(blocks, ggml_type, rows, columns).map_err(from_files)?;
    Ok(QuantizedInner::F16(Arc::new(tensor)))
}
