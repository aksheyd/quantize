//! Python input conversion.

use numpy::{PyArray1, PyArrayDyn, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyTuple};

use crate::error::{length_mismatch, InvalidBitsError, InvalidBlockError};

const MASKED_VALUES: &str = "values can't be a masked array, since its mask would be ignored; fill in the masked values first, like values.filled(0)";
const CODES_TYPE: &str = "codes must be a 1-D signed integer array or a sequence of int; packed Quantized.codes is uint8 and must not be passed here — use unpacked_codes";
const PACKED_CODES_TYPE: &str = "codes must be a 1-D uint8 array, like Quantized.codes";
const OUT_TYPE: &str = "out must be a writable C-contiguous native-endian float32 array";
const OUT_CONTIG: &str = "out must be writable and C-contiguous";

fn is_native_dtype(arr: &Bound<'_, PyUntypedArray>) -> PyResult<bool> {
    arr.dtype().getattr("isnative")?.extract()
}

fn dtype_kind(arr: &Bound<'_, PyUntypedArray>) -> PyResult<String> {
    arr.dtype().getattr("kind")?.extract()
}

fn type_name(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    obj.get_type().name().map(|n| n.to_string())
}

/// Read a 1-D or 2-D array of real numbers as `f32` values row after row.
/// Returns the values and the shape they came in.
pub fn as_f32_array(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<f32>, Vec<usize>)> {
    read_f32(obj, &[1, 2], "a 1-D or 2-D array")
}

/// Read a 1-D array of real numbers as `f32` values.
pub fn as_f32_values(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    read_f32(obj, &[1], "a 1-D array").map(|(values, _)| values)
}

/// Anything that `numpy.asarray` turns into an array works, like a list or a
/// PyTorch tensor, as long as it holds real numbers and has one of
/// `dimensions`, which `wanted` describes.
fn read_f32(
    obj: &Bound<'_, PyAny>,
    dimensions: &[usize],
    wanted: &str,
) -> PyResult<(Vec<f32>, Vec<usize>)> {
    let numpy = obj.py().import("numpy")?;
    if obj.is_instance(&numpy.getattr("ma")?.getattr("MaskedArray")?)? {
        return Err(PyTypeError::new_err(MASKED_VALUES));
    }
    let converted = numpy.call_method1("asarray", (obj,))?;
    let array = converted.cast::<PyUntypedArray>()?;
    if !matches!(dtype_kind(array)?.as_str(), "b" | "i" | "u" | "f") {
        return Err(PyTypeError::new_err(format!(
            "values must be {wanted} of real numbers, got {}",
            describe(obj, array)?
        )));
    }
    if !dimensions.contains(&array.ndim()) {
        return Err(PyValueError::new_err(format!(
            "values must be {wanted}, got {}",
            describe(obj, array)?
        )));
    }
    let float32 = numpy.getattr("float32")?;
    let contiguous = numpy.call_method1("ascontiguousarray", (array, float32))?;
    let typed = contiguous.cast::<PyArrayDyn<f32>>()?;
    let readonly = typed.try_readonly()?;
    Ok((readonly.as_slice()?.to_vec(), array.shape().to_vec()))
}

/// `obj`'s type, and the shape and dtype that `numpy.asarray` gave it, such
/// as `torch.Tensor with shape (2, 3, 4) and dtype float32`.
fn describe(obj: &Bound<'_, PyAny>, array: &Bound<'_, PyUntypedArray>) -> PyResult<String> {
    let shape = PyTuple::new(obj.py(), array.shape())?;
    Ok(format!(
        "{} with shape {} and dtype {}",
        obj.get_type().fully_qualified_name()?,
        shape.repr()?,
        array.dtype().str()?
    ))
}

/// Read matmul input: one vector of shape `(columns,)`, or a batch of shape
/// `(batch, columns)` flattened row after row. Returns the values and, for a
/// batch, its size.
pub fn as_f32_matmul_values(
    obj: &Bound<'_, PyAny>,
    columns: usize,
) -> PyResult<(Vec<f32>, Option<usize>)> {
    let (values, shape) = as_f32_array(obj)?;
    let input_columns = shape[shape.len() - 1];
    if input_columns != columns {
        return Err(length_mismatch(
            "each vector in values",
            columns,
            input_columns,
        ));
    }
    let batch = (shape.len() == 2).then_some(shape[0]);
    Ok((values, batch))
}

pub fn as_i32_codes(obj: &Bound<'_, PyAny>) -> PyResult<Vec<i32>> {
    let name = type_name(obj)?;
    if name == "memoryview" || name == "array" {
        return Err(PyTypeError::new_err(CODES_TYPE));
    }
    if let Ok(arr) = obj.cast::<PyUntypedArray>() {
        if arr.ndim() != 1 {
            return Err(PyTypeError::new_err(CODES_TYPE));
        }
        let kind = dtype_kind(arr)?;
        if kind != "i" {
            return Err(PyTypeError::new_err(CODES_TYPE));
        }
        if !is_native_dtype(arr)? {
            return Err(PyTypeError::new_err(CODES_TYPE));
        }
        let numpy = obj.py().import("numpy")?;
        let int64 = numpy.getattr("int64")?;
        let converted = arr.as_any().call_method1("astype", (int64,))?;
        let typed = converted.cast::<PyArray1<i64>>()?;
        let readonly = typed.try_readonly()?;
        let mut out = Vec::with_capacity(readonly.len());
        for &value in readonly.as_array().iter() {
            let code = i32::try_from(value)
                .map_err(|_| PyOverflowError::new_err("code is outside the i32 range"))?;
            out.push(code);
        }
        return Ok(out);
    }
    if obj.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(CODES_TYPE));
    }
    let ints: Vec<i64> = obj
        .extract()
        .map_err(|_| PyTypeError::new_err(CODES_TYPE))?;
    ints.into_iter()
        .map(|value| {
            i32::try_from(value)
                .map_err(|_| PyOverflowError::new_err("code is outside the i32 range"))
        })
        .collect()
}

/// Read packed codes: a 1-D uint8 array, like `Quantized.codes` returns.
pub fn as_packed_codes(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    let codes = obj
        .cast::<PyArray1<u8>>()
        .map_err(|_| PyTypeError::new_err(PACKED_CODES_TYPE))?;
    Ok(codes.try_readonly()?.as_array().to_vec())
}

/// Borrow `out` for writing, after checking it has exactly `shape`.
pub fn as_writable_f32_out<'py>(
    obj: &Bound<'py, PyAny>,
    shape: &[usize],
) -> PyResult<numpy::PyReadwriteArrayDyn<'py, f32>> {
    let arr = obj
        .cast::<PyArrayDyn<f32>>()
        .map_err(|_| PyTypeError::new_err(OUT_TYPE))?;
    if !is_native_dtype(arr.as_untyped())? {
        return Err(PyTypeError::new_err(OUT_TYPE));
    }
    let flags = arr.getattr("flags")?;
    let c_contiguous: bool = flags.getattr("c_contiguous")?.extract()?;
    let writeable: bool = flags.getattr("writeable")?.extract()?;
    if !c_contiguous || !writeable {
        return Err(PyValueError::new_err(OUT_CONTIG));
    }
    if arr.shape() != shape {
        let len = shape.iter().product();
        if arr.len() != len {
            return Err(length_mismatch("out", len, arr.len()));
        }
        let expected = PyTuple::new(obj.py(), shape)?;
        return Err(PyValueError::new_err(format!(
            "out must have shape {}, got {}",
            expected.repr()?,
            arr.getattr("shape")?.repr()?
        )));
    }
    arr.try_readwrite()
        .map_err(|_| PyValueError::new_err(OUT_CONTIG))
}

/// Read a `bits` argument. A negative width is as far out of range as 1 or
/// 17, so it raises `InvalidBitsError` too, instead of `OverflowError`.
pub fn bits_argument(obj: &Bound<'_, PyAny>) -> PyResult<u32> {
    let bits: i64 = obj.extract()?;
    u32::try_from(bits).map_err(|_| PyErr::new::<InvalidBitsError, _>(bits))
}

/// Read a `block` argument. A negative size raises `InvalidBlockError`, as 0
/// does, instead of `OverflowError`.
pub fn block_argument(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    let block: i64 = obj.extract()?;
    usize::try_from(block).map_err(|_| PyErr::new::<InvalidBlockError, _>(block))
}
