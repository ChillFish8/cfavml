//! A aligned buffer type for `ndarray` that aligns its memory to the cache line (64 bytes)
//!
//! This can provide several advantages where performance is critical as it generally greatly
//! improves the performance of memory fetching and pre-fetching at the cost of potentially
//! wasting a bit of memory.
//!
//!
//!


use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::{mem, ptr};


/// An aligned Array representation.
///
/// *Don’t use this type directly—use the type alias
/// [`AlignedArray`](crate::AlignedArray) for the array type!*
///
/// This type aligns its memory to 64 bytes.
pub struct AlignedOwnedRepr<A> {
    /// The internal vector holding the buffer aligned to 64 bytes.
    buffer: Vec<AlignTo64>,
    /// The actual length of the array/elements it contains.
    len: usize,
    /// The inner array type.
    inner_type: PhantomData<A>,
}

impl<A: Debug> Debug for AlignedOwnedRepr<A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.as_slice())
    }
}

impl<A> AlignedOwnedRepr<A> {
    /// Copies the content of the given slice to the aligned buffer.
    ///
    /// Due to the alignment restrictions, we cannot ever take an owned vec
    /// directly.
    pub(crate) fn copy_from_slice(data: &[A]) -> Self {
        let mut slf = Self::with_size(data.len());
        slf.extend_from_slice(data);
        slf
    }

    /// Creates a new empty buffer with capacity for the given size.
    ///
    /// This will immediately
    pub(crate) fn with_size(size: usize) -> Self {
        // If our entity fits within the 64 byte block or not.
        let minimum_capacity = if mem::size_of::<A>() > 64 {
            let multiply_by = (mem::size_of::<A>() / 64) + 1;
            (size * multiply_by) + multiply_by
        } else {
            (size / (64 / mem::size_of::<A>())) + 1
        };

        let mut buffer = Vec::with_capacity(minimum_capacity);

        Self {
            buffer,
            len: 0,
            inner_type: PhantomData,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Returns a slice view of the array.
    fn as_slice(&self) -> &[A] {
        unsafe {
            std::slice::from_raw_parts(self.buffer.as_ptr().cast(), self.len)
        }
    }

    /// Returns a mutable slice to the vector as type `Vec<A>`.
    ///
    /// This uses the length the actual capacity of the allocated
    /// blocks, rather than the true length.
    fn as_raw_slice_mut(&mut self) -> &mut [A] {
        unsafe {
            std::slice::from_raw_parts_mut(self.buffer.as_mut_ptr().cast(), self.capacity())
        }
    }

    fn get_next_write_pos(&self) -> usize {
        self.len * mem::size_of::<A>()
    }

    pub(crate) fn push(&mut self, item: A) {
        self.reserve_capacity(1);

        let start_offset = self.get_next_write_pos();
        let slice = self.as_raw_slice_mut();
        unsafe { *slice.get_unchecked_mut(start_offset) = item };

        self.len += 1;
    }

    pub(crate) fn extend_from_slice(&mut self, slice: &[A]) {
        self.reserve_capacity(slice.len());

        let start_offset = self.get_next_write_pos();

        unsafe {
            let buffer_ptr = self.buffer.as_mut_ptr();
            ptr::copy_nonoverlapping(
                slice.as_ptr(),
                buffer_ptr.add(start_offset).cast(),
                slice.len(),
            );
        };

        self.len += slice.len();
    }

    pub(crate) fn reserve_capacity(&mut self, num_extra: usize) {
        let spare_capacity = self.spare_capacity();
        let minimum_required_capacity = mem::size_of::<A>() * num_extra;

        if spare_capacity >= minimum_required_capacity {
            return;
        }

        let missing_capacity = minimum_required_capacity - spare_capacity;
        let num_blocks_to_alloc = (missing_capacity / 64) + 1;

        self.buffer.extend((0..num_blocks_to_alloc).map(|_| AlignTo64::default()));
    }

    pub(crate) fn capacity(&self) -> usize {
        (self.buffer.len() * 64) / mem::size_of::<A>()
    }

    pub(crate) fn spare_capacity(&self) -> usize {
        let capacity = self.capacity();
        capacity - (mem::size_of::<A>() * self.len)
    }
}


#[derive(Copy, Clone)]
#[repr(C, align(64))]
/// A type which is forcefully aligned to 64 bytes.
///
/// The internal type stored may not be a `u8` but we use
/// this type as our core representation in memory.
struct AlignTo64([u8; 64]);

impl Default for AlignTo64 {
    fn default() -> Self {
        Self([0; 64])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_buffer_create() {
        let buffer = AlignedOwnedRepr::<u8>::with_size(5);
        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.capacity(), 0);

        let buffer = AlignedOwnedRepr::<u64>::with_size(5);
        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.capacity(), 0);

        let buffer = AlignedOwnedRepr::<[u64; 32]>::with_size(5);
        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.capacity(), 0);
    }

    #[test]
    fn test_aligned_buffer_mutate() {
        let mut buffer = AlignedOwnedRepr::<u8>::with_size(5);
        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.capacity(), 0);

        buffer.push(123);
        assert_eq!(buffer.len(), 1);
        assert_eq!(buffer.capacity(), 64);

        let mut buffer = AlignedOwnedRepr::<[u64; 32]>::with_size(5);
        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.capacity(), 0);

        dbg!(mem::size_of::<[u64; 32]>());
        buffer.push([32432; 32]);
        assert_eq!(buffer.len(), 1);
        assert_eq!(buffer.capacity(), 1);
    }
}