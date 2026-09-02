//! Python input conversion.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBool;

use crate::error::length_mismatch;

const VALUES_TYPE: &str = "values must be a 1-D float32 array or a sequence of floats";
const VALUES_ENDIAN: &str = "values must be native-endian";
const MATMUL_MATRIX: &str =
    "2-D values must be a C-contiguous native-endian float32 array of shape (batch, columns)";
const CODES_TYPE: &str = "codes must be a 1-D signed integer array or a sequence of int; packed Quantized.codes is uint8 and must not be passed here — use unpacked_codes";
const OUT_TYPE: &str = "out must be a 1-D writable C-contiguous native-endian float32 array";
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

pub fn as_f32_values(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    let name = type_name(obj)?;
    if name == "memoryview" || name == "array" {
        return Err(PyTypeError::new_err(VALUES_TYPE));
    }
    if let Ok(arr) = obj.cast::<PyUntypedArray>() {
        if arr.ndim() != 1 {
            return Err(PyTypeError::new_err(VALUES_TYPE));
        }
        if dtype_kind(arr)? == "O" {
            return Err(PyTypeError::new_err(VALUES_TYPE));
        }
        if !is_native_dtype(arr)? {
            return Err(PyTypeError::new_err(VALUES_ENDIAN));
        }
        let numpy = obj.py().import("numpy")?;
        let float32 = numpy.getattr("float32")?;
        let converted = arr.as_any().call_method1("astype", (float32,))?;
        let typed = converted.cast::<PyArray1<f32>>()?;
        let readonly = typed.try_readonly()?;
        return Ok(readonly.as_array().iter().copied().collect());
    }
    obj.extract::<Vec<f32>>()
        .map_err(|_| PyTypeError::new_err(VALUES_TYPE))
}

/// Flatten 2-D C-contiguous float32 `(batch, columns)` so matmul can treat
/// it as `k * columns`. 1-D reuses `as_f32_values`; the `usize` is the
/// default column count (`len` or `shape[1]`).
pub fn as_f32_matmul_values(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<f32>, usize)> {
    if let Ok(arr) = obj.cast::<PyUntypedArray>() {
        if arr.ndim() == 2 {
            return flatten_c_contiguous_f32_matrix(arr);
        }
    }
    let values = as_f32_values(obj)?;
    let columns = values.len();
    Ok((values, columns))
}

fn flatten_c_contiguous_f32_matrix(arr: &Bound<'_, PyUntypedArray>) -> PyResult<(Vec<f32>, usize)> {
    let typed = arr
        .cast::<PyArray2<f32>>()
        .map_err(|_| PyTypeError::new_err(MATMUL_MATRIX))?;
    if !is_native_dtype(arr)? {
        return Err(PyTypeError::new_err(VALUES_ENDIAN));
    }
    if !arr.is_c_contiguous() {
        return Err(PyValueError::new_err(MATMUL_MATRIX));
    }
    let columns = arr.shape()[1];
    let readonly = typed.try_readonly()?;
    Ok((readonly.as_slice()?.to_vec(), columns))
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

pub fn as_writable_f32_out<'py>(
    obj: &Bound<'py, PyAny>,
    expected_len: usize,
) -> PyResult<numpy::PyReadwriteArray1<'py, f32>> {
    let arr = obj
        .cast::<PyArray1<f32>>()
        .map_err(|_| PyTypeError::new_err(OUT_TYPE))?;
    if arr.ndim() != 1 {
        return Err(PyTypeError::new_err(OUT_TYPE));
    }
    if !is_native_dtype(arr.as_untyped())? {
        return Err(PyTypeError::new_err(OUT_TYPE));
    }
    let flags = arr.getattr("flags")?;
    let c_contiguous: bool = flags.getattr("c_contiguous")?.extract()?;
    let writeable: bool = flags.getattr("writeable")?.extract()?;
    if !c_contiguous || !writeable {
        return Err(PyValueError::new_err(OUT_CONTIG));
    }
    if arr.len() != expected_len {
        return Err(length_mismatch(expected_len, arr.len()));
    }
    arr.try_readwrite()
        .map_err(|_| PyValueError::new_err(OUT_CONTIG))
}
