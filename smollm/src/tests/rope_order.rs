//! Checks q's and k's rows against the order llama.cpp's converter writes
//! them in, for floats and for quantized blocks.

use quantize::{Scheme, f16};
use quantize_files::Tensor;

use crate::saved_files::rope_order;

/// A matrix of 2 heads of 6 rows each, whose row `i` holds `i` in each of
/// its 32 columns.
fn numbered_rows() -> Tensor<f16> {
    let values = (0..12).flat_map(|row| [row as f32; 32]).collect();
    Tensor::Float {
        shape: vec![12, 32],
        values,
    }
}

#[test]
fn rope_order_is_llama_cpps() {
    let reordered = rope_order::to_llama_cpp(numbered_rows(), 2).unwrap();
    let Tensor::Float { values, .. } = &reordered else {
        panic!("floats stay floats");
    };
    let rows: Vec<f32> = values.chunks(32).map(|row| row[0]).collect();
    // The rows that `LlamaModel.permute(rows, 2, 2)`, in llama.cpp's
    // convert_hf_to_gguf.py, gives these.
    let llama_cpp_rows = [0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11];
    assert_eq!(rows, llama_cpp_rows.map(|row| row as f32));
    assert_eq!(
        rope_order::to_hugging_face(reordered, 2).unwrap(),
        numbered_rows()
    );
}

#[test]
fn quantized_rows_move_with_their_blocks() {
    let values: Vec<f32> = (0..8 * 64).map(|i| (i as f32 * 0.37).sin()).collect();
    let quantized = |values: &[f32]| {
        let mut matrix = Scheme::Q4_32.quantize::<f16>(values).unwrap();
        matrix.set_shape(8, 64).unwrap();
        Tensor::Quantized(matrix)
    };
    let floats = Tensor::Float {
        shape: vec![8, 64],
        values: values.clone(),
    };
    let Ok(Tensor::Float {
        values: reordered, ..
    }) = rope_order::to_llama_cpp(floats, 2)
    else {
        panic!("floats stay floats");
    };

    // Each block lies within a row, so moving the quantized rows gives the
    // blocks that quantizing the moved rows does, and moving them back gives
    // the matrix they came from.
    let moved = rope_order::to_llama_cpp(quantized(&values), 2).unwrap();
    assert_eq!(moved, quantized(&reordered));
    assert_eq!(
        rope_order::to_hugging_face(moved, 2).unwrap(),
        quantized(&values)
    );
}
