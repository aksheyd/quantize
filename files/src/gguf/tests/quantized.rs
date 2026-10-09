use std::collections::BTreeMap;
use std::num::FpCategory;

use half::f16;
use quantize::{Quantized, Scheme};

use super::{file, tensor_info};
use crate::Tensor;
use crate::gguf::read::from_bytes;
use crate::gguf::write::to_bytes;
use crate::gguf::{GgmlType, Metadata, from_ggml, to_ggml};

/// `count` weights spread like a layer's, by a normal distribution with
/// standard deviation 0.02, from `seed`, so that every run makes the same
/// ones.
fn weights(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut uniform = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32 + 0.5) / (1 << 24) as f32
    };
    let mut normal = move || {
        let radius = (-2.0 * uniform().ln()).sqrt();
        radius * (std::f32::consts::TAU * uniform()).cos()
    };
    (0..count).map(|_| 0.02 * normal()).collect()
}

/// 4 rows of 2 blocks each, built around the cases where two quantizers can
/// disagree, or a reader can go wrong.
fn edge_cases() -> Vec<f32> {
    let mut values = Vec::new();
    let mut block = |set: &[(usize, f32)], fill: f32| {
        let start = values.len();
        values.resize(start + 32, fill);
        for &(index, value) in set {
            values[start + index] = value;
        }
    };
    // Row 0: all zeros, then a block whose value farthest from zero is positive.
    block(&[], 0.0);
    block(&[(4, 1.0), (9, -0.5), (20, 0.3), (31, -0.3)], 0.1);
    // Row 1: -0.75 and 0.75 tie for farthest from zero, in either order.
    block(&[(0, -0.75), (5, 0.75), (17, 0.2)], -0.05);
    block(&[(0, 0.75), (5, -0.75), (17, 0.2)], -0.05);
    // Row 2: with -8 or 8 farthest, the 4-bit scale is 1 or -1, so these
    // land halfway between two codes.
    let halves = [
        (1, -2.5),
        (2, -1.5),
        (3, -0.5),
        (4, 0.5),
        (5, 1.5),
        (6, 2.5),
        (7, -3.5),
    ];
    block(&[&[(0, -8.0)][..], &halves].concat(), 0.25);
    block(&[&[(0, 8.0)][..], &halves].concat(), 0.25);
    // Row 3: values so small that the f16 scales are subnormal.
    values.extend(weights(64, 3).iter().map(|weight| weight * 5e-5));
    values
}

/// The values that a quantized tensor decodes to, as bit patterns, so that
/// -0 differs from 0.
fn decoded_bits(tensor: &Tensor<f16>) -> Vec<u32> {
    let Tensor::Quantized(quantized) = tensor else {
        panic!("{tensor:?} isn't quantized");
    };
    bits(&quantized.dequantize())
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn q4_32_and_q8_32_matrices_round_trip_through_q4_0_and_q8_0() {
    let mut outliers = weights(4 * 96, 2);
    for weight in outliers.iter_mut().step_by(37) {
        *weight *= 20.0;
    }
    let matrices = [
        ("weights", 8, 64, weights(8 * 64, 1)),
        ("outliers", 4, 96, outliers),
        ("edge cases", 4, 64, edge_cases()),
    ];
    let mut tensors = BTreeMap::new();
    for (name, rows, columns, values) in matrices {
        for scheme in [Scheme::Q4_32, Scheme::Q8_32] {
            let mut quantized = scheme.quantize::<f16>(&values).unwrap();
            quantized.set_shape(rows, columns).unwrap();
            tensors.insert(format!("{name}, {scheme}"), Tensor::Quantized(quantized));
        }
    }
    let (_, read_back) = from_bytes(&to_bytes(&Metadata::new(), &tensors).unwrap()).unwrap();
    assert_eq!(read_back, tensors);
    for (name, tensor) in &tensors {
        assert_eq!(
            decoded_bits(&read_back[name]),
            decoded_bits(tensor),
            "{name}"
        );
    }

    // The edge cases are really there: subnormal scales, and in 8 bits, the
    // code -128, which ggml's own quantizer never writes.
    let Tensor::Quantized(edge_cases) = &tensors["edge cases, symmetric(bits=8, block=32)"] else {
        unreachable!()
    };
    let scales = edge_cases.scales();
    assert!(
        scales
            .iter()
            .any(|scale| scale.classify() == FpCategory::Subnormal)
    );
    assert!(edge_cases.unpacked_codes().contains(&-128));
}

#[test]
fn to_ggml_gives_the_blocks_a_file_holds_and_from_ggml_reads_them_back() {
    for (scheme, ggml_type) in [
        (Scheme::Q4_32, GgmlType::Q4_0),
        (Scheme::Q8_32, GgmlType::Q8_0),
    ] {
        let mut quantized = scheme.quantize::<f16>(&edge_cases()).unwrap();
        quantized.set_shape(4, 64).unwrap();
        let (blocks, to_type) = to_ggml(&quantized).unwrap();
        assert_eq!(to_type, ggml_type);
        assert_eq!(ggml_type.name().parse::<GgmlType>().unwrap(), ggml_type);
        assert_eq!(blocks.len(), 4 * 2 * ggml_type.block_bytes());

        // A file of this one tensor ends with its data: these blocks, padded
        // to 32 bytes.
        let tensors = BTreeMap::from([("x".to_string(), Tensor::Quantized(quantized.clone()))]);
        let file = to_bytes(&Metadata::new(), &tensors).unwrap();
        let data = &file[file.len() - blocks.len().next_multiple_of(32)..];
        assert_eq!(data[..blocks.len()], blocks);

        assert_eq!(from_ggml(&blocks, ggml_type, 4, 64).unwrap(), quantized);
    }
}

#[test]
fn q4_0_and_q8_0_blocks_decode_as_the_module_docs_lay_them_out() {
    // A Q4_0 block with scale 0.5, where byte j holds j in its low 4 bits
    // and 15 - j in its high 4, so codes j - 8, then 7 - j.
    let mut q4_0 = f16::from_f32(0.5).to_le_bytes().to_vec();
    q4_0.extend((0..16).map(|j| j | (15 - j) << 4));
    let q4_0_codes = (0..16).map(|j| j - 8).chain((0..16).map(|j| 7 - j));
    let q4_0_values: Vec<f32> = q4_0_codes.map(|code| code as f32 * 0.5).collect();
    // A Q8_0 block with scale -0.25 and codes from -128 to 127.
    let q8_0_codes: Vec<i8> = (0..32).map(|i| (i * 255 / 31 - 128) as i8).collect();
    let mut q8_0 = f16::from_f32(-0.25).to_le_bytes().to_vec();
    q8_0.extend(q8_0_codes.iter().map(|&code| code as u8));
    let q8_0_values: Vec<f32> = q8_0_codes.iter().map(|&code| code as f32 * -0.25).collect();

    // Each tensor is 1 row of 32, padded to 32 bytes.
    let mut data = q4_0;
    data.resize(32, 0);
    data.extend(q8_0);
    data.resize(96, 0);
    let infos = [
        tensor_info("q4_0", &[32, 1], 2, 0),
        tensor_info("q8_0", &[32, 1], 8, 32),
    ];
    let bytes = file(&[], &infos, &data);
    let (_, tensors) = from_bytes(&bytes).unwrap();
    assert_eq!(decoded_bits(&tensors["q4_0"]), bits(&q4_0_values));
    assert_eq!(decoded_bits(&tensors["q8_0"]), bits(&q8_0_values));
    // Written back, the blocks are the same bytes.
    assert_eq!(to_bytes(&Metadata::new(), &tensors).unwrap(), bytes);
}

#[test]
fn a_quantized_tensor_of_any_dimensions_reads_as_a_matrix_of_its_rows() {
    // Q8_0 block `b` has scale b + 1 and codes -16 to 15.
    let blocks: Vec<u8> = (0..6)
        .flat_map(|b| {
            let codes = (0..32).map(|i| (i - 16) as i8 as u8);
            f16::from_f32(b as f32 + 1.0)
                .to_le_bytes()
                .into_iter()
                .chain(codes)
        })
        .collect();
    let values: Vec<f32> = (0..6)
        .flat_map(|b| (0..32).map(move |i| (i - 16) as f32 * (b as f32 + 1.0)))
        .collect();
    // Innermost first: 3 × 2 rows of 32, and 1 row of 192.
    let shapes: [(&[u64], usize, usize); 2] = [(&[32, 2, 3], 6, 32), (&[192], 1, 192)];
    for (dimensions, rows, columns) in shapes {
        let bytes = file(&[], &[tensor_info("x", dimensions, 8, 0)], &blocks);
        let (_, tensors) = from_bytes(&bytes).unwrap();
        let Tensor::Quantized(matrix) = &tensors["x"] else {
            panic!("{dimensions:?} didn't read as quantized");
        };
        assert_eq!(matrix.shape(), Some((rows, columns)), "{dimensions:?}");
        assert_eq!(matrix.dequantize(), values, "{dimensions:?}");
        assert!(matches!(matrix, Quantized::Symmetric { block: 32, .. }));
    }
}
