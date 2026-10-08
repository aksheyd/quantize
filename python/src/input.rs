//! Python input conversion.

use numpy::{
    PyArray1, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyOverflowError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyTuple};

use crate::error::{InvalidBitsError, InvalidBlockError, length_mismatch};

const CODES_TYPE: &str = "codes must be a 1-D signed integer array or a sequence of int; packed Quantized.codes is uint8 and must not be passed here — use unpacked_codes";
const PACKED_CODES_TYPE: &str = "codes must be a 1-D uint8 array, like Quantized.codes";
const BYTES_TYPE: &str = "data must be bytes, like to_bytes returns, or a 1-D uint8 array";
const OUT_TYPE: &str = "out must be a float32 numpy array or pytorch tensor";
const OUT_CONTIG: &str = "out must be writable and C-contiguous";
const OUT_OVERLAPS_INPUTS: &str = "out can't share memory with inputs";
const OUT_IN_USE: &str =
    "out is in use by another call, like one on another thread; give each call its own out";

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
pub fn as_f32_array<'py>(
    obj: &Bound<'py, PyAny>,
) -> PyResult<(PyReadonlyArrayDyn<'py, f32>, Vec<usize>)> {
    read_f32(obj, "values", &[1, 2], "a 1-D or 2-D array")
}

/// Read a 1-D array of real numbers as `f32` values.
pub fn as_f32_values<'py>(obj: &Bound<'py, PyAny>) -> PyResult<PyReadonlyArrayDyn<'py, f32>> {
    read_f32(obj, "values", &[1], "a 1-D array").map(|(values, _)| values)
}

/// Anything that `numpy.asarray` turns into an array works, like a list, and
/// so does any PyTorch tensor, as long as it holds real numbers and has one
/// of `dimensions`, which `wanted` describes. Errors call it `argument`. An
/// array that already holds C-contiguous float32 values is read where it is,
/// without a copy.
fn read_f32<'py>(
    obj: &Bound<'py, PyAny>,
    argument: &str,
    dimensions: &[usize],
    wanted: &str,
) -> PyResult<(PyReadonlyArrayDyn<'py, f32>, Vec<usize>)> {
    let numpy = obj.py().import("numpy")?;
    if obj.is_instance(&numpy.getattr("ma")?.getattr("MaskedArray")?)? {
        return Err(PyTypeError::new_err(format!(
            "{argument} can't be a masked array, since its mask would be ignored; fill in the masked values first, like {argument}.filled(0)"
        )));
    }
    let converted = numpy.call_method1("asarray", (readable_by_numpy(obj)?,))?;
    let array = converted.cast::<PyUntypedArray>()?;
    if !matches!(dtype_kind(array)?.as_str(), "b" | "i" | "u" | "f") {
        return Err(PyTypeError::new_err(format!(
            "{argument} must be {wanted} of real numbers, got {}",
            describe(obj, array)?
        )));
    }
    if !dimensions.contains(&array.ndim()) {
        return Err(PyValueError::new_err(format!(
            "{argument} must be {wanted}, got {}",
            describe(obj, array)?
        )));
    }
    let float32 = numpy.getattr("float32")?;
    let contiguous = numpy.call_method1("ascontiguousarray", (array, float32))?;
    let typed = contiguous.cast::<PyArrayDyn<f32>>()?;
    // Reading fails only while another call writes into the array.
    let readonly = typed.try_readonly().map_err(|_| {
        PyValueError::new_err(format!(
            "{argument} is being written by another call, like a matmul or dequantize with out= on another thread; read it once that call returns"
        ))
    })?;
    Ok((readonly, array.shape().to_vec()))
}

/// `obj`, or if it's a PyTorch tensor, a tensor that `numpy.asarray` reads:
/// detached, since NumPy refuses one that requires grad, and in float32 if
/// it holds floating-point numbers, since NumPy has no bfloat16.
fn readable_by_numpy<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if !is_torch_tensor(obj)? {
        return Ok(obj.clone());
    }
    let tensor = obj.call_method0("detach")?;
    if tensor.call_method0("is_floating_point")?.is_truthy()? {
        return tensor.call_method0("float");
    }
    Ok(tensor)
}

/// Whether `obj` is a PyTorch tensor. A program that passes one has imported
/// torch, so torch is looked up in `sys.modules` instead of imported, and
/// NumPy arrays, the usual input, skip even that.
fn is_torch_tensor(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    if obj.is_instance_of::<PyUntypedArray>() {
        return Ok(false);
    }
    let modules = obj.py().import("sys")?.getattr("modules")?;
    match modules.cast::<PyDict>()?.get_item("torch")? {
        Some(torch) if !torch.is_none() => obj.is_instance(&torch.getattr("Tensor")?),
        _ => Ok(false),
    }
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
pub fn as_f32_matmul_values<'py>(
    obj: &Bound<'py, PyAny>,
    columns: usize,
) -> PyResult<(PyReadonlyArrayDyn<'py, f32>, Option<usize>)> {
    let (values, shape) = read_f32(obj, "inputs", &[1, 2], "a 1-D or 2-D array")?;
    let input_columns = shape[shape.len() - 1];
    if input_columns != columns {
        return Err(length_mismatch(
            "each vector in inputs",
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
    read_uint8(obj, PACKED_CODES_TYPE)
}

/// Read block widths: a 1-D uint8 array, like `Quantized.block_bits`
/// returns, or a sequence of ints. The array is copied at once, since reading
/// it as a sequence would convert one NumPy scalar at a time.
pub fn as_block_bits(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    match obj.cast::<PyArray1<u8>>() {
        Ok(widths) => {
            let widths = widths.try_readonly()?;
            Ok(widths
                .as_array()
                .iter()
                .map(|&bits| u32::from(bits))
                .collect())
        }
        Err(_) => obj.extract(),
    }
}

/// Read the bytes that `to_bytes` saved: `bytes` or another bytes-like
/// object, or a 1-D uint8 array.
pub fn as_bytes(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    match PyBuffer::<u8>::get(obj) {
        Ok(buffer) => buffer.to_vec(obj.py()),
        Err(_) => read_uint8(obj, BYTES_TYPE),
    }
}

/// Read a 1-D uint8 array, or anything that `numpy.asarray` turns into one,
/// like a PyTorch tensor. Anything else raises `TypeError(message)`.
fn read_uint8(obj: &Bound<'_, PyAny>, message: &'static str) -> PyResult<Vec<u8>> {
    let array = obj.py().import("numpy")?.call_method1("asarray", (obj,))?;
    let bytes = array
        .cast::<PyArray1<u8>>()
        .map_err(|_| PyTypeError::new_err(message))?;
    Ok(bytes.try_readonly()?.as_array().to_vec())
}

/// `out` as a NumPy array to write into: itself, or for a PyTorch tensor,
/// the array that `out.numpy()` gives, which shares the tensor's memory. A
/// tensor must be float32 and on the CPU, and can't require grad, since
/// writing into it would go around autograd.
fn out_array<'py>(out: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if !is_torch_tensor(out)? {
        return Ok(out.clone());
    }
    let device = out.getattr("device")?.str()?;
    if device.to_cow()? != "cpu" {
        return Err(PyTypeError::new_err(format!(
            "out must be a tensor on the cpu, got one on {device}"
        )));
    }
    if out.getattr("requires_grad")?.is_truthy()? {
        return Err(PyTypeError::new_err(
            "out can't be a tensor that requires grad",
        ));
    }
    if out.getattr("dtype")?.str()?.to_cow()? != "torch.float32" {
        return Err(out_type_error(out)?);
    }
    out.call_method0("numpy")
}

/// A `TypeError` that says what `out` must be, and what it is: its type,
/// and its dtype if it has one, like `numpy.ndarray with dtype float64`.
fn out_type_error(out: &Bound<'_, PyAny>) -> PyResult<PyErr> {
    let mut description = out.get_type().fully_qualified_name()?.to_string();
    if let Ok(dtype) = out.getattr("dtype") {
        description += &format!(" with dtype {}", dtype.str()?);
    }
    let message = format!("{OUT_TYPE}, got {description}");
    Ok(PyTypeError::new_err(message))
}

/// Borrow `out`, a NumPy array or PyTorch tensor, for writing, after
/// checking it has exactly `shape` and shares no memory with `inputs`, the
/// values the call reads while it writes `out`.
pub fn as_writable_f32_out<'py>(
    obj: &Bound<'py, PyAny>,
    shape: &[usize],
    inputs: &[f32],
) -> PyResult<numpy::PyReadwriteArrayDyn<'py, f32>> {
    let array = out_array(obj)?;
    let Ok(arr) = array.cast::<PyArrayDyn<f32>>() else {
        return Err(out_type_error(obj)?);
    };
    if !is_native_dtype(arr.as_untyped())? {
        return Err(out_type_error(obj)?);
    }
    let flags = arr.getattr("flags")?;
    let c_contiguous: bool = flags.getattr("c_contiguous")?.extract()?;
    let writeable: bool = flags.getattr("writeable")?.extract()?;
    if !c_contiguous || !writeable {
        return Err(PyValueError::new_err(OUT_CONTIG));
    }
    check_shape(obj.py(), "out", shape, arr.shape())?;
    // This comes before the borrow, which fails on an `out` that `inputs`
    // was read from in place, as if another call held it.
    if shares_memory(arr, inputs) {
        return Err(PyValueError::new_err(OUT_OVERLAPS_INPUTS));
    }
    // Its flags are checked above, so this fails only while another call
    // reads or writes it.
    arr.try_readwrite()
        .map_err(|_| PyValueError::new_err(OUT_IN_USE))
}

/// Whether `out` and `inputs`, both C-contiguous float32, share memory.
/// Each fills one unbroken run of bytes, so they share memory exactly when
/// the runs overlap. The memory is compared, not the arrays, since two
/// arrays over one buffer, like `t.numpy()` and the array read from a
/// tensor `t`, are separate objects. Comparing addresses keeps the GIL,
/// which `numpy.may_share_memory` gives up.
fn shares_memory(out: &Bound<'_, PyArrayDyn<f32>>, inputs: &[f32]) -> bool {
    let out_start = out.data().addr();
    let out_end = out_start + out.len() * size_of::<f32>();
    let inputs_start = inputs.as_ptr().addr();
    let inputs_end = inputs_start + size_of_val(inputs);
    // The bytes both fill run from the later start to the earlier end.
    out_start.max(inputs_start) < out_end.min(inputs_end)
}

/// Check that `argument`, which came in with shape `got`, has exactly the
/// tensor's `shape`. The wrong number of values raises `LengthMismatchError`,
/// and the right number in another shape, like a transposed matrix, raises
/// `ValueError`.
pub fn check_shape(py: Python<'_>, argument: &str, shape: &[usize], got: &[usize]) -> PyResult<()> {
    if got == shape {
        return Ok(());
    }
    let len: usize = shape.iter().product();
    let got_len: usize = got.iter().product();
    if got_len != len {
        return Err(length_mismatch(argument, len, got_len));
    }
    Err(PyValueError::new_err(format!(
        "{argument} must have shape {}, got {}",
        PyTuple::new(py, shape)?.repr()?,
        PyTuple::new(py, got)?.repr()?
    )))
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
