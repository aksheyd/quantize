//! Bit-packed signed integer codes.
//!
//! 4-bit and 8-bit paths are specialized; other widths use a general
//! bit-buffer.

use crate::params::{assert_bits_in_range, largest_code, smallest_code};

/// Packed signed codes plus the bit width they were written with.
///
/// Codes are stored as `bits`-wide two's-complement fields, packed LSB-first
/// into a `Vec<u8>`.
#[derive(Clone, PartialEq, Eq)]
pub struct Packed {
    bytes: Vec<u8>,
    bits: u32,
    len: usize,
}

impl Packed {
    /// Pack `codes` using `bits` bits each.
    ///
    /// Panics if `bits` is outside `2..=16`. Each code must fit in `bits`,
    /// from [`smallest_code`] to [`largest_code`]. Debug builds check that;
    /// release builds keep a code's low `bits` bits, which read back as
    /// another code.
    pub fn from_i32s(codes: &[i32], bits: u32) -> Self {
        assert_bits_in_range(bits);
        debug_assert!(
            codes
                .iter()
                .all(|code| (smallest_code(bits)..=largest_code(bits)).contains(code)),
            "every code must fit in {bits} bits"
        );
        let mut p = Self {
            bytes: vec![0u8; nbytes(codes.len(), bits)],
            bits,
            len: codes.len(),
        };
        match bits {
            8 => pack_i8(&mut p.bytes, codes),
            4 => pack_i4(&mut p.bytes, codes),
            _ => pack_general(&mut p.bytes, codes, bits),
        }
        p
    }

    /// Number of codes.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether there are no codes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Bit width of each code.
    #[inline]
    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// Raw packed bytes.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Wrap already-packed bytes. `bytes` must hold `len` codes of `bits`.
    ///
    /// Panics if `bits` is outside `2..=16`.
    pub fn from_raw(bytes: Vec<u8>, bits: u32, len: usize) -> Self {
        assert_bits_in_range(bits);
        Self { bytes, bits, len }
    }

    /// Unpack into `out`, which must be at least [`len`](Self::len) long.
    pub fn unpack_into(&self, out: &mut [i32]) {
        debug_assert!(out.len() >= self.len);
        match self.bits {
            8 => unpack_i8(&self.bytes, out, self.len),
            4 => unpack_i4(&self.bytes, out, self.len),
            _ => unpack_general(&self.bytes, out, self.len, self.bits),
        }
    }

    /// Unpack `n` codes of width `bits` from a raw byte slice.
    ///
    /// Panics if `bits` is outside `2..=16`.
    pub(crate) fn unpack_slice(bytes: &[u8], bits: u32, out: &mut [i32], n: usize) {
        assert_bits_in_range(bits);
        match bits {
            8 => unpack_i8(bytes, out, n),
            4 => unpack_i4(bytes, out, n),
            _ => unpack_general(bytes, out, n, bits),
        }
    }

    /// Unpack the codes from `start` to `start + out.len()`, which may begin
    /// partway through a byte.
    pub(crate) fn unpack_range(&self, start: usize, out: &mut [i32]) {
        for (index, slot) in (start..).zip(out) {
            *slot = read_code(&self.bytes, index, self.bits);
        }
    }

    /// Read code `index` alone, without unpacking the others.
    #[inline]
    pub(crate) fn code(&self, index: usize) -> i32 {
        read_code(&self.bytes, index, self.bits)
    }
}

/// A one-line summary, like `Packed { bits: 4, len: 64, .. }`. The bytes are
/// left out, since one layer holds millions of codes.
impl core::fmt::Debug for Packed {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Packed")
            .field("bits", &self.bits)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

#[inline]
pub(crate) fn nbytes(len: usize, bits: u32) -> usize {
    (len * bits as usize).div_ceil(8)
}

fn pack_i8(bytes: &mut [u8], codes: &[i32]) {
    for (b, &q) in bytes.iter_mut().zip(codes) {
        *b = q as i8 as u8;
    }
}

fn unpack_i8(bytes: &[u8], out: &mut [i32], n: usize) {
    for i in 0..n {
        out[i] = bytes[i] as i8 as i32;
    }
}

fn pack_i4(bytes: &mut [u8], codes: &[i32]) {
    for (i, chunk) in codes.chunks(2).enumerate() {
        let lo = (chunk[0] as u8) & 0x0F;
        let hi = chunk.get(1).copied().unwrap_or(0) as u8 & 0x0F;
        bytes[i] = lo | (hi << 4);
    }
}

fn unpack_i4(bytes: &[u8], out: &mut [i32], n: usize) {
    for i in 0..n {
        let byte = bytes[i / 2];
        let nib = if i.is_multiple_of(2) {
            byte & 0x0F
        } else {
            byte >> 4
        };
        out[i] = (((nib as i8) << 4) >> 4) as i32;
    }
}

fn pack_general(bytes: &mut [u8], codes: &[i32], bits: u32) {
    for (i, &q) in codes.iter().enumerate() {
        write_code(bytes, i, bits, q);
    }
}

fn unpack_general(bytes: &[u8], out: &mut [i32], n: usize, bits: u32) {
    for (i, slot) in out.iter_mut().enumerate().take(n) {
        *slot = read_code(bytes, i, bits);
    }
}

fn write_code(bytes: &mut [u8], index: usize, bits: u32, q: i32) {
    let mask = (1u32 << bits) - 1;
    let val = (q as u32) & mask;
    let bit = index * bits as usize;
    let byte = bit / 8;
    let off = bit % 8;
    let wide = (val as u64) << off;
    bytes[byte] |= wide as u8;
    if off + bits as usize > 8 {
        bytes[byte + 1] |= (wide >> 8) as u8;
    }
    if off + bits as usize > 16 {
        bytes[byte + 2] |= (wide >> 16) as u8;
    }
}

// Every loop that reads codes calls this once per code, so it must be inlined
// into each of them. It doesn't check `bits`, so callers check it first, as
// `Packed` does when it's built.
#[inline(always)]
pub(crate) fn read_code(bytes: &[u8], index: usize, bits: u32) -> i32 {
    let bit = index * bits as usize;
    let byte = bit / 8;
    let off = bit % 8;
    let mut wide = bytes[byte] as u32 >> off;
    if off + bits as usize > 8 {
        wide |= (bytes[byte + 1] as u32) << (8 - off);
    }
    if off + bits as usize > 16 {
        wide |= (bytes[byte + 2] as u32) << (16 - off);
    }
    let mask = (1u32 << bits) - 1;
    let u = wide & mask;
    let sign = 1u32 << (bits - 1);
    if u & sign != 0 {
        (u | !mask) as i32
    } else {
        u as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_bit_roundtrip_preserves_signed_codes() {
        let codes = [-8, -1, 0, 7, 3, -4];
        let p = Packed::from_i32s(&codes, 4);
        let mut out = [0i32; 6];
        p.unpack_into(&mut out);
        assert_eq!(out, codes);
        assert_eq!(p.as_bytes().len(), 3);
    }

    #[test]
    fn five_bit_roundtrip_preserves_signed_codes() {
        let codes = [-16, -1, 0, 15, 7];
        let p = Packed::from_i32s(&codes, 5);
        let mut out = [0i32; 5];
        p.unpack_into(&mut out);
        assert_eq!(out, codes);
    }

    #[test]
    #[should_panic(expected = "bits must be in 2..=16")]
    fn packing_at_seventeen_bits_panics() {
        Packed::from_i32s(&[0], 17);
    }

    #[test]
    #[should_panic(expected = "bits must be in 2..=16")]
    fn unpacking_at_one_bit_panics() {
        Packed::unpack_slice(&[0], 1, &mut [0], 1);
    }

    #[test]
    #[should_panic(expected = "bits must be in 2..=16")]
    fn wrapping_bytes_at_seventeen_bits_panics() {
        Packed::from_raw(vec![0; 3], 17, 1);
    }

    // Without the check, 9, -9, and 100 would come back as -7, 7, and 4.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "every code must fit in 4 bits")]
    fn a_code_too_wide_for_its_bits_panics_in_debug_builds() {
        Packed::from_i32s(&[9, -9, 100, 7, -8], 4);
    }

    #[test]
    fn debug_prints_a_summary_instead_of_every_byte() {
        let p = Packed::from_i32s(&[3; 64], 4);
        assert_eq!(format!("{p:?}"), "Packed { bits: 4, len: 64, .. }");
    }

    #[test]
    fn unpack_range_starts_partway_through_a_byte() {
        let codes = [-8, -1, 0, 7, 3, -4];
        for bits in [4, 5, 8] {
            let p = Packed::from_i32s(&codes, bits);
            let mut out = [0i32; 3];
            p.unpack_range(1, &mut out);
            assert_eq!(out, codes[1..4], "{bits} bits");
        }
    }
}
