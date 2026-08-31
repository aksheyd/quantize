//! Bit-packed signed integer codes.
//!
//! Codes are stored as `bits`-wide two's-complement fields, packed LSB-first
//! into a `Vec<u8>`. 4-bit and 8-bit paths are specialized; other widths use
//! a general bit-buffer.

/// Packed signed codes plus the bit width they were written with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Packed {
    bytes: Vec<u8>,
    bits: u32,
    len: usize,
}

impl Packed {
    /// Pack `codes` using `bits` bits each.
    pub fn from_i32s(codes: &[i32], bits: u32) -> Self {
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
    pub fn from_raw(bytes: Vec<u8>, bits: u32, len: usize) -> Self {
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
    pub fn unpack_slice(bytes: &[u8], bits: u32, out: &mut [i32], n: usize) {
        match bits {
            8 => unpack_i8(bytes, out, n),
            4 => unpack_i4(bytes, out, n),
            _ => unpack_general(bytes, out, n, bits),
        }
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

fn read_code(bytes: &[u8], index: usize, bits: u32) -> i32 {
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
}
