//! A cursor over a file's bytes, which reads the numbers and strings that
//! come before the data.

use crate::error::{Error, invalid};

/// The bytes of a whole file, and how far into them the reader has got.
pub(super) struct Reader<'a> {
    pub(super) bytes: &'a [u8],
    pub(super) position: usize,
}

impl<'a> Reader<'a> {
    /// The next `count` bytes.
    pub(super) fn take(&mut self, count: usize) -> Result<&'a [u8], Error> {
        let end = self.position.checked_add(count);
        let Some(taken) = end.and_then(|end| self.bytes.get(self.position..end)) else {
            return Err(invalid(format!(
                "the file ends in the middle of its header, after {} bytes",
                self.bytes.len()
            )));
        };
        self.position += count;
        Ok(taken)
    }

    /// The next `N` bytes, which hold a number.
    pub(super) fn bytes<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        Ok(self.take(N)?.try_into().expect("take gives back N bytes"))
    }

    pub(super) fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }

    pub(super) fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }

    /// A length, a dimension, or an offset, stored as a `u64`. Where a
    /// `usize` is smaller and can't hold it, it becomes `usize::MAX`, more
    /// than any file holds, so reading what it measures fails.
    pub(super) fn size(&mut self) -> Result<usize, Error> {
        Ok(usize::try_from(self.u64()?).unwrap_or(usize::MAX))
    }

    /// A string: its length in bytes, then its UTF-8 bytes.
    pub(super) fn string(&mut self) -> Result<String, Error> {
        let start = self.position;
        let length = self.size()?;
        let text = self.take(length)?;
        String::from_utf8(text.to_vec())
            .map_err(|_| invalid(format!("the string at byte {start} isn't UTF-8")))
    }
}
