use std::collections::BTreeMap;

use half::f16;
use quantize::Scheme;

use super::super::read::from_bytes;
use super::super::write::to_bytes;
use super::super::{Metadata, Value, read};
use super::{entry, file, float, temporary_path, tensor_info, values};
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
    let cases: [(Vec<Vec<u8>>, usize, &str); 7] = [
        (
            vec![tensor_info("x", &[32], 2, 0)],
            0,
            r#"tensor "x" has ggml type 2, but only F32 (0), F16 (1), and BF16 (30) are read"#,
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
    let quantized = Scheme::Q4_32.quantize::<f16>(&values(32)).unwrap();
    let tensor_cases = [
        (
            wrong_shape,
            r#"tensor "x" has 5 values, which don't fit its shape [2, 3]"#,
        ),
        (
            Tensor::Quantized(quantized),
            r#"tensor "x" is quantized, but only float tensors are written to gguf"#,
        ),
    ];
    for (tensor, message) in tensor_cases {
        let tensors = BTreeMap::from([("x".to_string(), tensor)]);
        let error = to_bytes(&Metadata::new(), &tensors).unwrap_err();
        assert_eq!(error.to_string(), message);
    }

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
fn a_file_that_does_not_open_gives_its_path() {
    let path = temporary_path("missing");
    let error = read(&path).unwrap_err();
    assert!(matches!(error, Error::Io { path: ref failed, .. } if *failed == path));
    assert!(error.to_string().starts_with(&*path.to_string_lossy()));
}
