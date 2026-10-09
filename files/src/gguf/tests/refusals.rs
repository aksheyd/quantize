use std::collections::BTreeMap;

use half::f16;
use quantize::Scheme;

use super::{entry, file, float, temporary_path, tensor_info, values};
use crate::gguf::read::from_bytes;
use crate::gguf::write::to_bytes;
use crate::gguf::{Metadata, Value, read};
use crate::{Error, Tensor};

fn refusal(bytes: &[u8]) -> String {
    from_bytes(bytes).unwrap_err().to_string()
}

#[test]
fn a_file_that_is_not_little_endian_gguf_version_3_is_refused() {
    let valid = file(&[], &[], &[]);
    let with_version = |version: [u8; 4]| [&valid[..4], &version, &valid[8..]].concat();
    let cases = [
        (
            [b"GGUX", &valid[4..]].concat(),
            "the file doesn't start with GGUF",
        ),
        (
            with_version(2_u32.to_le_bytes()),
            "the file is gguf version 2, but only version 3 is read",
        ),
        (
            with_version(3_u32.to_be_bytes()),
            "the file is big-endian gguf, but only little-endian is read",
        ),
        (
            valid[..20].to_vec(),
            "the file ends in the middle of its header, after 20 bytes",
        ),
    ];
    for (bytes, message) in cases {
        assert_eq!(refusal(&bytes), message);
    }
}

#[test]
fn each_broken_rule_of_the_metadata_is_refused_with_what_is_wrong() {
    let u8_entry = entry("k", 0, &[1]);
    let not_utf8 = [&1_u64.to_le_bytes()[..], &[0xFF]].concat();
    // An array whose one element is an array, and so on: each gives its
    // elements' type, 9 for an array, and its length, 1.
    let nested_65_deep = [&9_u32.to_le_bytes()[..], &1_u64.to_le_bytes()]
        .concat()
        .repeat(64);
    let cases: [(Vec<Vec<u8>>, &str); 9] = [
        (
            // A string 100 bytes long, in a file of 64.
            vec![entry("k", 8, &100_u64.to_le_bytes())],
            "the file ends in the middle of its header, after 64 bytes",
        ),
        (
            vec![entry("k", 13, &[])],
            r#"metadata "k" has value type 13, which gguf doesn't define"#,
        ),
        (
            vec![entry(
                "k",
                9,
                &[&13_u32.to_le_bytes()[..], &1_u64.to_le_bytes()].concat(),
            )],
            r#"metadata "k" has value type 13, which gguf doesn't define"#,
        ),
        (
            vec![entry("k", 7, &[2])],
            r#"metadata "k" is a bool, but its byte is 2, not 0 or 1"#,
        ),
        (
            // The value starts after the 24-byte header, the 9-byte key, and
            // the 4-byte type.
            vec![entry("k", 8, &not_utf8)],
            "the string at byte 37 isn't UTF-8",
        ),
        (
            vec![entry("k", 9, &nested_65_deep)],
            r#"metadata "k" nests arrays more than 64 deep"#,
        ),
        (
            vec![u8_entry.clone(), u8_entry],
            r#"metadata "k" appears twice"#,
        ),
        (
            vec![entry("general.alignment", 10, &32_u64.to_le_bytes())],
            "general.alignment must be a U32 that is a power of two, not U64(32)",
        ),
        (
            vec![entry("general.alignment", 4, &48_u32.to_le_bytes())],
            "general.alignment must be a U32 that is a power of two, not U32(48)",
        ),
    ];
    for (entries, message) in cases {
        assert_eq!(refusal(&file(&entries, &[], &[])), message);
    }
}

#[test]
fn each_broken_rule_of_the_tensors_is_refused_with_what_is_wrong() {
    let one = |name: &str, offset: u64| tensor_info(name, &[1], 0, offset);
    let sixteen = tensor_info("x", &[16], 0, 0);
    let cases: [(Vec<Vec<u8>>, usize, &str); 9] = [
        (
            vec![tensor_info("x", &[32], 3, 0)],
            0,
            r#"tensor "x" has ggml type 3, but only F32 (0), F16 (1), Q4_0 (2), Q8_0 (8), and BF16 (30) are read"#,
        ),
        (
            vec![tensor_info("x", &[48], 2, 0)],
            0,
            r#"tensor "x" is Q4_0 with rows of 48 values, which don't split into blocks of 32"#,
        ),
        (
            vec![tensor_info("x", &[0, 2], 8, 0)],
            0,
            r#"tensor "x" is Q8_0 with rows of no values, which quantize's tensors can't hold"#,
        ),
        (
            vec![tensor_info("x", &[u64::MAX, 2], 0, 0)],
            0,
            r#"tensor "x" has shape [2, 18446744073709551615], which holds too many bytes to count"#,
        ),
        (
            vec![one("x", 0), one("y", 8)],
            64,
            r#"tensor "y" starts at byte 8 of the data, which isn't a multiple of the alignment, 32"#,
        ),
        // "x" holds 64 bytes, so "y" can't start at 32.
        (
            vec![sixteen.clone(), one("y", 32)],
            96,
            r#"tensor "y" starts at byte 32 of the data, but the tensors before it, padded to the alignment, end at byte 64"#,
        ),
        (
            vec![one("x", 32)],
            64,
            r#"tensor "x" starts at byte 32 of the data, but the tensors before it, padded to the alignment, end at byte 0"#,
        ),
        (
            vec![one("x", 0), one("x", 32)],
            64,
            r#"tensor "x" appears twice"#,
        ),
        (
            vec![sixteen],
            32,
            r#"tensor "x"'s data runs past the end of the file, at byte 96"#,
        ),
    ];
    for (tensor_infos, data_length, message) in cases {
        let bytes = file(&[], &tensor_infos, &vec![0; data_length]);
        assert_eq!(refusal(&bytes), message);
    }
}

#[test]
fn writing_refuses_what_the_format_cannot_hold() {
    let wrong_shape = Tensor::Float {
        shape: vec![2, 3],
        values: vec![0.0; 5],
    };
    let tensors = BTreeMap::from([("x".to_string(), wrong_shape)]);
    let error = to_bytes(&Metadata::new(), &tensors).unwrap_err();
    assert_eq!(
        error.to_string(),
        r#"tensor "x" has 5 values, which don't fit its shape [2, 3]"#
    );

    let metadata_cases = [
        (
            "general.alignment",
            Value::U32(0),
            "general.alignment must be a U32 that is a power of two, not U32(0)",
        ),
        (
            "k",
            Value::Array(Vec::new()),
            r#"metadata "k" is an empty array, which has no element type to write"#,
        ),
        (
            "k",
            Value::Array(vec![Value::U8(1), Value::I8(1)]),
            r#"metadata "k" is an array whose elements don't all have one type"#,
        ),
    ];
    let tensors = BTreeMap::from([("x".to_string(), float(&[1]))]);
    for (key, value, message) in metadata_cases {
        let metadata = Metadata::from([(key.to_string(), value)]);
        let error = to_bytes(&metadata, &tensors).unwrap_err();
        assert_eq!(error.to_string(), message);
    }
}

#[test]
fn writing_refuses_what_ggml_would_not_load() {
    let long_name = "x".repeat(64);
    let tensor_cases = [
        (
            long_name.clone(),
            float(&[1]),
            format!(
                "tensor {long_name:?}'s name is 64 bytes long, but ggml loads names of at most 63"
            ),
        ),
        (
            "x".to_string(),
            float(&[1, 1, 1, 1, 1]),
            r#"tensor "x" has 5 dimensions, but ggml loads at most 4"#.to_string(),
        ),
    ];
    for (name, tensor, message) in tensor_cases {
        let tensors = BTreeMap::from([(name, tensor)]);
        let error = to_bytes(&Metadata::new(), &tensors).unwrap_err();
        assert_eq!(error.to_string(), message);
    }
    // A byte shorter and a dimension fewer are within ggml's limits.
    let at_the_limits = BTreeMap::from([("x".repeat(63), float(&[1, 1, 1, 1]))]);
    assert!(to_bytes(&Metadata::new(), &at_the_limits).is_ok());

    let nested = Value::Array(vec![Value::Array(vec![Value::I32(1)])]);
    let metadata = Metadata::from([("k".to_string(), nested)]);
    let error = to_bytes(&metadata, &BTreeMap::new()).unwrap_err();
    assert_eq!(
        error.to_string(),
        r#"metadata "k" is an array of arrays, which ggml doesn't load"#
    );
}

#[test]
fn writing_refuses_quantized_tensors_that_q4_0_and_q8_0_cannot_hold() {
    let matrix = |scheme: Scheme, rows: usize, columns: usize| {
        let mut quantized = scheme.quantize::<f16>(&values(rows * columns)).unwrap();
        quantized.set_shape(rows, columns).unwrap();
        quantized
    };
    let adaptive = Scheme::Adaptive {
        block: 32,
        tolerance: 0.01,
    };
    let cases = [
        (
            matrix(Scheme::Asymmetric { bits: 4, block: 32 }, 2, 32),
            "is asymmetric, but Q4_0 and Q8_0 are symmetric",
        ),
        (
            matrix(adaptive, 2, 32),
            "is adaptive, but Q4_0 and Q8_0 give every block one width",
        ),
        (
            matrix(Scheme::Symmetric { bits: 6, block: 32 }, 2, 32),
            "has 6-bit codes, but Q4_0 and Q8_0 have 4-bit and 8-bit ones",
        ),
        (
            matrix(Scheme::Symmetric { bits: 8, block: 64 }, 2, 64),
            "has blocks of 64, but Q4_0 and Q8_0 have blocks of 32",
        ),
        (
            Scheme::Q4_32.quantize::<f16>(&values(64)).unwrap(),
            "has no shape, but Q4_0 and Q8_0 blocks run along a matrix's rows, which set_shape records",
        ),
        (
            matrix(Scheme::Q8_32, 2, 48),
            "has rows of 48 values, but Q4_0 and Q8_0 split each row into blocks of 32",
        ),
    ];
    for (quantized, why) in cases {
        let tensors = BTreeMap::from([("x".to_string(), Tensor::Quantized(quantized))]);
        let error = to_bytes(&Metadata::new(), &tensors).unwrap_err();
        let message = format!(r#"tensor "x" {why}; safetensors keeps any quantized tensor"#);
        assert_eq!(error.to_string(), message);
    }
}

#[test]
fn a_file_that_does_not_open_gives_its_path() {
    let path = temporary_path("missing");
    let error = read(&path).unwrap_err();
    assert!(matches!(error, Error::Io { path: ref failed, .. } if *failed == path));
    assert!(error.to_string().starts_with(&*path.to_string_lossy()));
}
