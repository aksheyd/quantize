//! Hot loops, split the same way the schemes are: reduce, then 4-bit, 8-bit,
//! then the slow general path.

mod block;
mod i4;
mod i8;
mod reduce;

pub(crate) use block::{
    dequant_asym_into, dequant_sym_into, dot_asym, dot_sym, quantize_asym_block,
    quantize_sym_packed,
};
pub(crate) use i4::{dequant_i4_blocks, dot_i4_blocks};
pub(crate) use i8::{dequant_i8_blocks, dot_i8_blocks};
pub(crate) use reduce::min_max;
