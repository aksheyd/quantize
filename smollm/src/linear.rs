//! A linear layer: the model's one kind of matrix multiply, with its weights
//! kept as plain floats or quantized.

use quantize::{Quantized, f16};
use quantize_files::Tensor;

/// A `rows × columns` matrix of weights, which turns an input of `columns`
/// values into `rows` outputs, each one the input's dot product with a row,
/// as PyTorch's `Linear` does.
pub enum Linear {
    /// Plain f32 weights, row after row.
    Float { columns: usize, values: Vec<f32> },
    /// Quantized weights, which `Quantized::matmul` multiplies straight from
    /// their codes, without decoding the matrix first.
    Quantized(Quantized<f16>),
}

impl Linear {
    /// The matrix `name` from a model file, which must be `rows × columns`.
    pub fn new(
        name: &str,
        tensor: Tensor<f16>,
        rows: usize,
        columns: usize,
    ) -> Result<Self, String> {
        let shape = match &tensor {
            Tensor::Float { shape, .. } => shape.clone(),
            Tensor::Quantized(weights) => weights
                .shape()
                .map_or(vec![], |(rows, columns)| vec![rows, columns]),
        };
        if shape != [rows, columns] {
            return Err(format!(
                "{name} has shape {shape:?}, but the config makes it [{rows}, {columns}]"
            ));
        }
        Ok(match tensor {
            Tensor::Float { values, .. } => Self::Float { columns, values },
            Tensor::Quantized(weights) => Self::Quantized(weights),
        })
    }

    /// Multiply the matrix by one input of `columns` values.
    pub fn multiply(&self, input: &[f32]) -> Vec<f32> {
        match self {
            Self::Float { columns, values } => values
                .chunks_exact(*columns)
                .map(|row| dot(row, input))
                .collect(),
            Self::Quantized(weights) => weights.matmul(input).expect("new() checked the shape"),
        }
    }

    /// Row `row` of the matrix. The embedding table looks up a token this way.
    pub fn row(&self, row: usize) -> Vec<f32> {
        match self {
            Self::Float { columns, values } => values[row * columns..(row + 1) * columns].to_vec(),
            Self::Quantized(weights) => {
                let (_, columns) = weights.shape().expect("new() checked the shape");
                let mut values = vec![0.0; columns];
                weights
                    .dequantize_row_into(row, &mut values)
                    .expect("the row is in the matrix");
                values
            }
        }
    }
}

const LANES: usize = 16;

/// Multiply two slices element by element and add up the products.
///
/// With one running total, the compiler must add the products one at a time,
/// in order. Sixteen separate totals let it add them side by side in SIMD
/// registers, so the f32 model runs about as fast as plain Rust allows.
pub fn dot(left: &[f32], right: &[f32]) -> f32 {
    let (left_chunks, left_rest) = left.as_chunks::<LANES>();
    let (right_chunks, right_rest) = right.as_chunks::<LANES>();
    let mut totals = [0.0_f32; LANES];
    for (left_chunk, right_chunk) in left_chunks.iter().zip(right_chunks) {
        for ((total, a), b) in totals.iter_mut().zip(left_chunk).zip(right_chunk) {
            *total += a * b;
        }
    }
    let rest: f32 = left_rest.iter().zip(right_rest).map(|(a, b)| a * b).sum();
    totals.iter().sum::<f32>() + rest
}
