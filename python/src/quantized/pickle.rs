use half::{bf16, f16};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};
use quantize::{Packed, Quantized, Scale};

use super::inner::{with_inner, QuantizedInner};
use crate::scale::PyScale;

const PICKLE_TAG: i32 = 1;
const PICKLE_LEN: usize = 10;

trait ScaleBits: Scale {
    const WIDTH: usize;

    fn write_le(self, output: &mut Vec<u8>);
    fn read_le(bytes: &[u8]) -> Option<Self>;
}

impl ScaleBits for f32 {
    const WIDTH: usize = 4;

    fn write_le(self, output: &mut Vec<u8>) {
        output.extend_from_slice(&self.to_le_bytes());
    }

    fn read_le(bytes: &[u8]) -> Option<Self> {
        Some(f32::from_le_bytes(bytes.try_into().ok()?))
    }
}

impl ScaleBits for f16 {
    const WIDTH: usize = 2;

    fn write_le(self, output: &mut Vec<u8>) {
        output.extend_from_slice(&self.to_bits().to_le_bytes());
    }

    fn read_le(bytes: &[u8]) -> Option<Self> {
        Some(f16::from_bits(u16::from_le_bytes(bytes.try_into().ok()?)))
    }
}

impl ScaleBits for bf16 {
    const WIDTH: usize = 2;

    fn write_le(self, output: &mut Vec<u8>) {
        output.extend_from_slice(&self.to_bits().to_le_bytes());
    }

    fn read_le(bytes: &[u8]) -> Option<Self> {
        Some(bf16::from_bits(u16::from_le_bytes(bytes.try_into().ok()?)))
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Symmetric,
    Asymmetric,
    Adaptive,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Self::Symmetric => "symmetric",
            Self::Asymmetric => "asymmetric",
            Self::Adaptive => "adaptive",
        }
    }

    fn parse(name: &str) -> PyResult<Self> {
        match name {
            "symmetric" => Ok(Self::Symmetric),
            "asymmetric" => Ok(Self::Asymmetric),
            "adaptive" => Ok(Self::Adaptive),
            _ => Err(malformed()),
        }
    }
}

struct Parts {
    kind: Kind,
    block: usize,
    len: usize,
    codes: Vec<u8>,
    code_bits: u32,
    block_bits: Vec<u32>,
    scales: Vec<u8>,
    zero_points: Vec<u8>,
}

fn malformed() -> PyErr {
    PyValueError::new_err("malformed pickle state")
}

fn encode<S: ScaleBits>(values: &[S]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * S::WIDTH);
    for &value in values {
        value.write_le(&mut bytes);
    }
    bytes
}

fn decode<S: ScaleBits>(bytes: &[u8]) -> PyResult<Vec<S>> {
    if !bytes.len().is_multiple_of(S::WIDTH) {
        return Err(malformed());
    }
    bytes
        .chunks_exact(S::WIDTH)
        .map(|chunk| S::read_le(chunk).ok_or_else(malformed))
        .collect()
}

fn parts<S: ScaleBits>(quantized: &Quantized<S>) -> Parts {
    match quantized {
        Quantized::Symmetric {
            scales,
            codes,
            block,
            len,
        } => Parts {
            kind: Kind::Symmetric,
            block: *block,
            len: *len,
            codes: codes.as_bytes().to_vec(),
            code_bits: codes.bits(),
            block_bits: Vec::new(),
            scales: encode(scales),
            zero_points: Vec::new(),
        },
        Quantized::Asymmetric {
            scales,
            zero_points,
            codes,
            block,
            len,
        } => Parts {
            kind: Kind::Asymmetric,
            block: *block,
            len: *len,
            codes: codes.as_bytes().to_vec(),
            code_bits: codes.bits(),
            block_bits: Vec::new(),
            scales: encode(scales),
            zero_points: encode(zero_points),
        },
        Quantized::Adaptive {
            scales,
            zero_points,
            bytes,
            bits,
            block,
            len,
        } => Parts {
            kind: Kind::Adaptive,
            block: *block,
            len: *len,
            codes: bytes.clone(),
            code_bits: 0,
            block_bits: bits.clone(),
            scales: encode(scales),
            zero_points: encode(zero_points),
        },
    }
}

fn rebuild<S: ScaleBits>(parts: Parts) -> PyResult<Quantized<S>> {
    let scales = decode(&parts.scales)?;
    match parts.kind {
        Kind::Symmetric => Ok(Quantized::Symmetric {
            scales,
            codes: Packed::from_raw(parts.codes, parts.code_bits, parts.len),
            block: parts.block,
            len: parts.len,
        }),
        Kind::Asymmetric => Ok(Quantized::Asymmetric {
            scales,
            zero_points: decode(&parts.zero_points)?,
            codes: Packed::from_raw(parts.codes, parts.code_bits, parts.len),
            block: parts.block,
            len: parts.len,
        }),
        Kind::Adaptive => Ok(Quantized::Adaptive {
            scales,
            zero_points: decode(&parts.zero_points)?,
            bytes: parts.codes,
            bits: parts.block_bits,
            block: parts.block,
            len: parts.len,
        }),
    }
}

pub(super) fn pickle_state<'py>(
    py: Python<'py>,
    inner: &QuantizedInner,
) -> PyResult<Bound<'py, PyAny>> {
    let scale = inner.scale();
    let parts = with_inner!(inner, |quantized| parts(quantized));
    Ok(PyTuple::new(
        py,
        [
            PICKLE_TAG.into_pyobject(py)?.into_any(),
            parts.kind.name().into_pyobject(py)?.into_any(),
            scale.into_pyobject(py)?.into_any(),
            parts.block.into_pyobject(py)?.into_any(),
            parts.len.into_pyobject(py)?.into_any(),
            PyBytes::new(py, &parts.codes).into_any(),
            parts.code_bits.into_pyobject(py)?.into_any(),
            PyTuple::new(py, &parts.block_bits)?.into_any(),
            PyBytes::new(py, &parts.scales).into_any(),
            PyBytes::new(py, &parts.zero_points).into_any(),
        ],
    )?
    .into_any())
}

pub(super) fn from_pickle(state: Bound<'_, PyAny>) -> PyResult<QuantizedInner> {
    let state = state.cast::<PyTuple>().map_err(|_| malformed())?;
    if state.len() != PICKLE_LEN || state.get_item(0)?.extract::<i32>()? != PICKLE_TAG {
        return Err(malformed());
    }

    let kind = Kind::parse(&state.get_item(1)?.extract::<String>()?)?;
    let scale = state
        .get_item(2)?
        .extract::<PyScale>()
        .map_err(|_| malformed())?;
    let parts = Parts {
        kind,
        block: state.get_item(3)?.extract()?,
        len: state.get_item(4)?.extract()?,
        codes: state.get_item(5)?.extract()?,
        code_bits: state.get_item(6)?.extract()?,
        block_bits: state.get_item(7)?.extract()?,
        scales: state.get_item(8)?.extract()?,
        zero_points: state.get_item(9)?.extract()?,
    };
    match scale {
        PyScale::F32 => rebuild(parts).map(QuantizedInner::F32),
        PyScale::F16 => rebuild(parts).map(QuantizedInner::F16),
        PyScale::Bf16 => rebuild(parts).map(QuantizedInner::Bf16),
    }
}
