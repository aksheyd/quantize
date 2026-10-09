//! Metadata values: their types, the number that gguf gives each type, and
//! their bytes.

use super::reader::Reader;
use super::write::write_string;
use crate::error::{Error, invalid};

/// One metadata value, like a model's architecture, its number of layers, or
/// its tokenizer's vocabulary. Each variant's type is the number that a
/// file stores before a value of that type.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// Type 0.
    U8(u8),
    /// Type 1.
    I8(i8),
    /// Type 2.
    U16(u16),
    /// Type 3.
    I16(i16),
    /// Type 4.
    U32(u32),
    /// Type 5.
    I32(i32),
    /// Type 10.
    U64(u64),
    /// Type 11.
    I64(i64),
    /// Type 6.
    F32(f32),
    /// Type 12.
    F64(f64),
    /// Type 7: one byte, 0 or 1.
    Bool(bool),
    /// Type 8: UTF-8 text.
    String(String),
    /// Type 9: values that all have one type. The format lets them be arrays
    /// too, which [`read()`](super::read) takes, but
    /// [`write()`](super::write) refuses, since ggml's own reader refuses
    /// arrays of arrays, so llama.cpp couldn't load the file.
    Array(Vec<Value>),
}

/// How deep arrays can nest in a file that [`read()`](super::read) reads, so
/// that a file nesting them deeper can't overflow the stack.
const DEEPEST_NESTING: usize = 64;

impl Value {
    fn type_number(&self) -> u32 {
        match self {
            Self::U8(_) => 0,
            Self::I8(_) => 1,
            Self::U16(_) => 2,
            Self::I16(_) => 3,
            Self::U32(_) => 4,
            Self::I32(_) => 5,
            Self::F32(_) => 6,
            Self::Bool(_) => 7,
            Self::String(_) => 8,
            Self::Array(_) => 9,
            Self::U64(_) => 10,
            Self::I64(_) => 11,
            Self::F64(_) => 12,
        }
    }

    /// Read metadata entry `key`'s value: its type's number, then its bytes.
    pub(super) fn read(reader: &mut Reader, key: &str) -> Result<Self, Error> {
        let type_number = reader.u32()?;
        Self::read_of_type(reader, key, type_number, 0)
    }

    /// Read a value of type `type_number`, inside `depth` arrays.
    fn read_of_type(
        reader: &mut Reader,
        key: &str,
        type_number: u32,
        depth: usize,
    ) -> Result<Self, Error> {
        Ok(match type_number {
            0 => Self::U8(u8::from_le_bytes(reader.bytes()?)),
            1 => Self::I8(i8::from_le_bytes(reader.bytes()?)),
            2 => Self::U16(u16::from_le_bytes(reader.bytes()?)),
            3 => Self::I16(i16::from_le_bytes(reader.bytes()?)),
            4 => Self::U32(u32::from_le_bytes(reader.bytes()?)),
            5 => Self::I32(i32::from_le_bytes(reader.bytes()?)),
            6 => Self::F32(f32::from_le_bytes(reader.bytes()?)),
            7 => match reader.bytes()? {
                [0] => Self::Bool(false),
                [1] => Self::Bool(true),
                [byte] => {
                    return Err(invalid(format!(
                        "metadata {key:?} is a bool, but its byte is {byte}, not 0 or 1"
                    )));
                }
            },
            8 => Self::String(reader.string()?),
            9 if depth == DEEPEST_NESTING => {
                return Err(invalid(format!(
                    "metadata {key:?} nests arrays more than {DEEPEST_NESTING} deep"
                )));
            }
            9 => {
                let element_type = reader.u32()?;
                let mut elements = Vec::new();
                for _ in 0..reader.u64()? {
                    elements.push(Self::read_of_type(reader, key, element_type, depth + 1)?);
                }
                Self::Array(elements)
            }
            10 => Self::U64(u64::from_le_bytes(reader.bytes()?)),
            11 => Self::I64(i64::from_le_bytes(reader.bytes()?)),
            12 => Self::F64(f64::from_le_bytes(reader.bytes()?)),
            other => {
                return Err(invalid(format!(
                    "metadata {key:?} has value type {other}, which gguf doesn't define"
                )));
            }
        })
    }

    /// Write metadata entry `key`'s value: its type's number, then its bytes.
    pub(super) fn write(&self, key: &str, bytes: &mut Vec<u8>) -> Result<(), Error> {
        bytes.extend(self.type_number().to_le_bytes());
        self.write_without_type(key, bytes)
    }

    /// Write this value's bytes alone, as an array's elements are written.
    fn write_without_type(&self, key: &str, bytes: &mut Vec<u8>) -> Result<(), Error> {
        match self {
            Self::U8(number) => bytes.extend(number.to_le_bytes()),
            Self::I8(number) => bytes.extend(number.to_le_bytes()),
            Self::U16(number) => bytes.extend(number.to_le_bytes()),
            Self::I16(number) => bytes.extend(number.to_le_bytes()),
            Self::U32(number) => bytes.extend(number.to_le_bytes()),
            Self::I32(number) => bytes.extend(number.to_le_bytes()),
            Self::U64(number) => bytes.extend(number.to_le_bytes()),
            Self::I64(number) => bytes.extend(number.to_le_bytes()),
            Self::F32(number) => bytes.extend(number.to_le_bytes()),
            Self::F64(number) => bytes.extend(number.to_le_bytes()),
            Self::Bool(flag) => bytes.push(u8::from(*flag)),
            Self::String(text) => write_string(bytes, text),
            Self::Array(elements) => {
                // An array stores its elements' type once, so they must share
                // it, and an empty array has no element to take it from.
                let Some(first) = elements.first() else {
                    return Err(invalid(format!(
                        "metadata {key:?} is an empty array, which has no element type to write"
                    )));
                };
                if let Self::Array(_) = first {
                    return Err(invalid(format!(
                        "metadata {key:?} is an array of arrays, which ggml doesn't load"
                    )));
                }
                let element_type = first.type_number();
                if elements
                    .iter()
                    .any(|element| element.type_number() != element_type)
                {
                    return Err(invalid(format!(
                        "metadata {key:?} is an array whose elements don't all have one type"
                    )));
                }
                bytes.extend(element_type.to_le_bytes());
                bytes.extend((elements.len() as u64).to_le_bytes());
                for element in elements {
                    element.write_without_type(key, bytes)?;
                }
            }
        }
        Ok(())
    }
}
