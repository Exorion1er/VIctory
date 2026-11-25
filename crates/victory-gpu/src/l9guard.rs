//! AnyXboxX lot 9 fix, lot 8 port (`crate::watch`'s guard self-deadlock):
//! what must be said where the heap or a lock may be what stopped
//! everything: a fixed text on the stack and a write straight to the
//! standard error handle. Lot 9 has these in its stall watchdog
//! (`stall.rs`), which lot 8 does not have.

/// A line prefix the log can be searched for.
pub(crate) const TAG: &str = "VICTORY-GUARD";

#[link(name = "kernel32")]
unsafe extern "system" {
    fn GetStdHandle(which: u32) -> *mut std::ffi::c_void;
    fn WriteFile(
        file: *mut std::ffi::c_void,
        buffer: *const u8,
        bytes: u32,
        written: *mut u32,
        overlapped: *mut std::ffi::c_void,
    ) -> i32;
}

/// Writes `text` to the standard error handle in one call per 64 KiB,
/// past every lock of ours, of tracing's and of `std::io::stderr`'s.
pub(crate) fn raw(text: &str) {
    const STD_ERROR_HANDLE: u32 = -12i32 as u32;
    // SAFETY: a plain query.
    let handle = unsafe { GetStdHandle(STD_ERROR_HANDLE) };
    if handle.is_null() || handle as isize == -1 {
        return;
    }
    for chunk in text.as_bytes().chunks(1 << 16) {
        let mut written = 0;
        // SAFETY: a live buffer of that length; synchronous handle.
        unsafe {
            WriteFile(
                handle,
                chunk.as_ptr(),
                chunk.len() as u32,
                &mut written,
                std::ptr::null_mut(),
            )
        };
    }
}

/// A text of at most `N` bytes on the stack: writing past the end cuts it
/// (on a character boundary), never allocates.
pub(crate) struct Fixed<const N: usize> {
    buf: [u8; N],
    len: usize,
}

impl<const N: usize> Fixed<N> {
    pub(crate) const fn new() -> Self {
        Fixed { buf: [0; N], len: 0 }
    }

    pub(crate) fn as_str(&self) -> &str {
        // SAFETY: only whole characters are copied in (`write_str`).
        unsafe { std::str::from_utf8_unchecked(&self.buf[..self.len]) }
    }
}

impl<const N: usize> std::fmt::Write for Fixed<N> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let room = N - self.len;
        let mut take = s.len().min(room);
        while !s.is_char_boundary(take) {
            take -= 1;
        }
        self.buf[self.len..self.len + take].copy_from_slice(&s.as_bytes()[..take]);
        self.len += take;
        if take < s.len() { Err(std::fmt::Error) } else { Ok(()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fixed_text_is_cut_on_a_character_boundary() {
        let mut small = Fixed::<8>::new();
        assert!(std::fmt::Write::write_str(&mut small, "naïveté!").is_err());
        assert_eq!(small.as_str(), "naïvet");
    }
}
