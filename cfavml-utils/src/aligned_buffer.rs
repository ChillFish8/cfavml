use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::mem;
use std::ops::Deref;

#[derive(Clone)]
/// A buffer that stores the a set of items in a buffer aligned to 64 bytes.
///
/// WARNING:
/// This buffer is primarily designed for use within CFAVML, and simply assumes
/// that it is safe to cast the buffer of `[u8]` to `[T]`.
pub struct AlignedBuffer<T> {
    len: usize,
    allocated_size: usize,
    buffer: Box<[AlignedBytes]>,
    inner: PhantomData<T>,
}

impl<T: Copy> AlignedBuffer<T> {
    /// Creates a new aligned buffer with a capacity of `size` elements.
    ///
    /// This method asserts that some multiples of `T` fit within a single `64B` buffer.
    ///
    /// I.e. `T` is of size where `64 % size == 0`.
    ///
    /// # Safety
    ///
    /// The inner buffer is _always_ aligned to 64B, if a type is ever beyond that alignment
    /// this can become UB.
    pub unsafe fn zeroed(len: usize) -> Self {
        assert_eq!(
            64 % mem::size_of::<T>(),
            0,
            "Size of `T` must be able to fit within a 64B buffer some \
            multiple of times without a remainder."
        );

        let num_per_chunk = 64 / mem::size_of::<T>();
        let num_chunks = (len / num_per_chunk) + 1;

        let mut buffer = Vec::with_capacity(num_chunks);
        buffer.extend(std::iter::repeat(AlignedBytes::default()).take(num_chunks));

        let buffer = buffer.into_boxed_slice();

        Self {
            len,
            allocated_size: num_per_chunk * buffer.len(),
            buffer,
            inner: PhantomData,
        }
    }

    #[inline]
    /// The actual size of buffer allocation and the maximum number of items
    /// it can actually hold.
    pub fn allocated_size(&self) -> usize {
        self.allocated_size
    }

    #[inline]
    /// Copies a slice of values from `data` to the inner buffer.
    ///
    /// It is expected that `len(data) == len(self)`
    pub fn copy_from_slice(&mut self, data: &[T]) {
        let slice = self.as_mut_slice();
        slice.copy_from_slice(data);
    }

    #[inline]
    /// Returns the buffer as a borrowed slice of `T`.
    pub fn as_slice(&self) -> &[T] {
        let ptr = self.buffer.as_ptr();
        unsafe { std::slice::from_raw_parts(ptr.cast(), self.len) }
    }

    #[inline]
    /// Returns the buffer as a borrowed slice of `T`.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        let ptr = self.buffer.as_mut_ptr();
        ptr.cast()
    }

    #[inline]
    /// Returns the buffer as a borrowed slice of `T`.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        let ptr = self.buffer.as_mut_ptr();
        unsafe { std::slice::from_raw_parts_mut(ptr.cast(), self.len) }
    }
}

impl<T: Copy> Deref for AlignedBuffer<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: Copy + Debug> Debug for AlignedBuffer<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "AlignedBuffer({:?})", self.as_slice())
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C, align(64))]
struct AlignedBytes([u8; 64]);

impl Default for AlignedBytes {
    fn default() -> Self {
        Self([0; 64])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeroed_buffer() {
        let buf: AlignedBuffer<f32> = unsafe { AlignedBuffer::zeroed(0) };
        assert_eq!(buf.as_slice(), &[]);
        assert_eq!(buf.allocated_size(), 16);

        let buf: AlignedBuffer<i8> = unsafe { AlignedBuffer::zeroed(4) };
        assert_eq!(buf.as_slice(), &[0; 4]);
        assert_eq!(buf.allocated_size(), 64);

        let buf: AlignedBuffer<u16> = unsafe { AlignedBuffer::zeroed(128) };
        assert_eq!(buf.as_slice(), &[0; 128]);
        assert_eq!(buf.allocated_size(), 160);
    }

    #[test]
    fn test_buffer_write() {
        let mut buf: AlignedBuffer<f32> = unsafe { AlignedBuffer::zeroed(5) };
        assert_eq!(buf.as_slice(), &[0.0; 5]);
        assert_eq!(buf.allocated_size(), 16);

        buf.copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(buf.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }
}
