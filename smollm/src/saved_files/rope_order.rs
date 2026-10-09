//! llama.cpp's order for the rows of the query and key matrices.
//!
//! Rope rotates pairs of each head's values, and the two llamas pair them
//! differently: Hugging Face's pairs value `i` with value `i + head_dim / 2`,
//! as `attention.rs` does, and llama.cpp's pairs neighbors, `2i` with
//! `2i + 1`. A head's queries are its rows of the q matrix times the input,
//! one value a row, and its keys the same with k. So llama.cpp's converter
//! reorders each head's rows of q and k, putting row `i` at `2i` and row
//! `i + head_dim / 2` at `2i + 1`, and llama.cpp's rope then turns the same
//! pairs by the same angles as Hugging Face's. Attention's scores don't
//! change: a query and a key move alike, and a dot product adds up the same
//! products in any order.
//!
//! Rows move whole, and each block of 32 values lies within one row, so a
//! quantized matrix's rows move with their blocks' scales and codes as they
//! are. Quantizing the reordered floats would make the same blocks, but this
//! needs no floats: the file holds the very blocks quantize made, and moving
//! them back gives the matrix that was saved, bit for bit.

use quantize::{Packed, Quantized, f16};
use quantize_files::Tensor;

/// Put each head's rows of `tensor`, a q or k matrix of `heads` heads, in
/// llama.cpp's order.
pub fn to_llama_cpp(tensor: Tensor<f16>, heads: usize) -> Result<Tensor<f16>, String> {
    // Row `2i` of llama.cpp's head is row `i` of Hugging Face's, and row
    // `2i + 1` is row `i + head_dim / 2`.
    reorder_heads(tensor, heads, |row, head_dim| {
        row / 2 + row % 2 * (head_dim / 2)
    })
}

/// Put each head's rows back in Hugging Face's order.
pub fn to_hugging_face(tensor: Tensor<f16>, heads: usize) -> Result<Tensor<f16>, String> {
    // Row `i` of Hugging Face's head is row `2i` of llama.cpp's, and row
    // `i + head_dim / 2` is row `2i + 1`.
    reorder_heads(tensor, heads, |row, head_dim| {
        row % (head_dim / 2) * 2 + row / (head_dim / 2)
    })
}

/// Reorder the rows within each of `heads` heads of `tensor`, a matrix of
/// floats or of symmetric blocks that split its rows: row `row` of a head
/// becomes the head's row `source(row, head_dim)`.
fn reorder_heads(
    tensor: Tensor<f16>,
    heads: usize,
    source: fn(usize, usize) -> usize,
) -> Result<Tensor<f16>, String> {
    let rows = match &tensor {
        Tensor::Float { shape, .. } => shape.first().copied(),
        Tensor::Quantized(matrix) => matrix.shape().map(|(rows, _)| rows),
    };
    // Rope turns pairs, so each head has an even number of rows.
    let Some(rows) = rows.filter(|rows| *rows > 0 && rows % (2 * heads) == 0) else {
        return Err(format!(
            "a q or k matrix's rows must split into {heads} heads of an even number of rows"
        ));
    };
    let head_dim = rows / heads;
    let order: Vec<usize> = (0..rows)
        .map(|row| row - row % head_dim + source(row % head_dim, head_dim))
        .collect();
    match tensor {
        Tensor::Float { shape, values } => {
            let values = in_order(&values, values.len() / rows, &order);
            Ok(Tensor::Float { shape, values })
        }
        Tensor::Quantized(Quantized::Symmetric {
            scales,
            codes,
            block,
            len,
            columns: Some(columns),
        }) if columns % block == 0 => {
            let mut unpacked = vec![0; len];
            codes.unpack_into(&mut unpacked);
            Ok(Tensor::Quantized(Quantized::Symmetric {
                scales: in_order(&scales, columns / block, &order),
                codes: Packed::from_i32s(&in_order(&unpacked, columns, &order), codes.bits()),
                block,
                len,
                columns: Some(columns),
            }))
        }
        _ => Err("a q or k matrix must hold floats or symmetric blocks that split its rows".into()),
    }
}

/// `values` in runs of `run_length`, with run `order[i]` moved to place `i`.
fn in_order<T: Copy>(values: &[T], run_length: usize, order: &[usize]) -> Vec<T> {
    let runs = order
        .iter()
        .map(|&run| &values[run * run_length..][..run_length]);
    runs.flatten().copied().collect()
}
