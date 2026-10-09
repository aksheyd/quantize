use std::collections::BTreeMap;

use half::f16;
use quantize::Scheme;

use super::{file, float, temporary_path, values};
use crate::safetensors::read;
use crate::safetensors::read::from_bytes;
use crate::safetensors::write::to_bytes;
use crate::{Error, Tensor};

fn refusal(bytes: &[u8]) -> String {
    from_bytes::<f32>(bytes).unwrap_err().to_string()
}

#[test]
fn a_file_too_short_for_its_header_is_refused() {
    let short = refusal(&[1, 0, 0]);
    assert_eq!(short, "the file is too short to hold its header's length");
    let cut_off = refusal(&[&100_u64.to_le_bytes()[..], b"{}"].concat());
    assert_eq!(
        cut_off,
        "the header should be 100 bytes long, but the file has 2 after its length"
    );
}

#[test]
fn each_broken_rule_of_the_header_is_refused_with_what_is_wrong() {
    let cases: [(&str, &[u8], &str); 13] = [
        (" {}", &[], "the header doesn't start with {"),
        (
            "{,}",
            &[],
            "the header isn't a JSON object: key must be a string at line 1 column 2",
        ),
        (
            r#"{"__metadata__": {"format": 1}}"#,
            &[],
            "__metadata__ must map strings to strings",
        ),
        (
            r#"{"x": {"dtype": "F32", "shape": [1]}}"#,
            &[0; 4],
            r#"tensor "x" needs a dtype, a shape, and two data_offsets"#,
        ),
        (
            r#"{"x": {"dtype": "I64", "shape": [1], "data_offsets": [0, 8]}}"#,
            &[0; 8],
            r#"tensor "x" is I64, but only F32, F16, BF16, and quantize's U8 tensors are read"#,
        ),
        (
            r#"{"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}"#,
            &[0; 4],
            r#"tensor "x"'s data_offsets [0, 4] don't hold F32 values of shape [2]"#,
        ),
        (
            r#"{"x": {"dtype": "F32", "shape": [0], "data_offsets": [4, 0]}}"#,
            &[0; 4],
            r#"tensor "x"'s data_offsets [4, 0] don't hold F32 values of shape [0]"#,
        ),
        (
            r#"{"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}"#,
            &[0; 4],
            r#"tensor "x"'s data_offsets [0, 8] run past the end of the data, at byte 4"#,
        ),
        (
            r#"{"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
                "y": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}}"#,
            &[0; 8],
            r#"tensor "y" starts at byte 4 of the data, inside the tensor before it"#,
        ),
        (
            r#"{"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                "y": {"dtype": "F32", "shape": [1], "data_offsets": [8, 12]}}"#,
            &[0; 12],
            "no tensor holds the 4 bytes of the data from byte 4",
        ),
        (
            r#"{"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}"#,
            &[0; 6],
            "no tensor holds the 2 bytes of the data from byte 4",
        ),
        // The format allows each name once, and the first "x" is lost.
        (
            r#"{"x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                "x": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}}"#,
            &[0; 8],
            "no tensor holds the 4 bytes of the data from byte 0",
        ),
        (
            r#"{"x": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}}"#,
            b"QNTX",
            r#"tensor "x" is U8, but doesn't start with QNTZ, so it isn't a tensor that quantize saved"#,
        ),
    ];
    for (header, data, message) in cases {
        assert_eq!(refusal(&file(header, data)), message, "{header}");
    }
}

#[test]
fn a_quantized_tensor_must_load_with_the_scale_type_asked_for() {
    let saved = Scheme::Q8_32
        .quantize::<f16>(&values(8))
        .unwrap()
        .to_bytes();
    let header = |length: usize| {
        format!(r#"{{"x": {{"dtype": "U8", "shape": [{length}], "data_offsets": [0, {length}]}}}}"#)
    };
    let as_f32 = refusal(&file(&header(saved.len()), &saved));
    assert_eq!(
        as_f32,
        r#"tensor "x": the tensor was saved with f16 scales, not f32"#
    );
    let cut_off = refusal(&file(&header(8), &saved[..8]));
    assert_eq!(
        cut_off,
        r#"tensor "x": malformed tensor: the bytes end early"#
    );
}

#[test]
fn writing_refuses_what_the_format_cannot_hold() {
    let wrong_shape = Tensor::<f32>::Float {
        shape: vec![2, 3],
        values: vec![0.0; 5],
    };
    let tensors = BTreeMap::from([("x".to_string(), wrong_shape)]);
    let error = to_bytes(&tensors).unwrap_err();
    assert_eq!(
        error.to_string(),
        r#"tensor "x" has 5 values, which don't fit its shape [2, 3]"#
    );

    let tensors = BTreeMap::from([("__metadata__".to_string(), float::<f32>(&[1]))]);
    let error = to_bytes(&tensors).unwrap_err();
    assert_eq!(error.to_string(), "a tensor can't be named __metadata__");
}

#[test]
fn a_file_that_does_not_open_gives_its_path() {
    let path = temporary_path("missing");
    let error = read::<f32>(&path).unwrap_err();
    assert!(matches!(error, Error::Io { path: ref failed, .. } if *failed == path));
    assert!(error.to_string().starts_with(&*path.to_string_lossy()));
}
