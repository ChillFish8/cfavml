//! The core buffer trait used to represent writeable buffers.
//!
//! This is used to work around the fact that the various CFAVML routines
//! support both uninitialized and initialized memory, which in Rust requires
//! either a `&mut [T]` or `&mut [MaybeUninit<T>]`.
use core::mem::MaybeUninit;

/// Represents a buffer that can only safely be written to.
///
/// When accessing the buffer pointer via this trait it is assumed that
/// the buffer will _never_ be read from and only every written to.
pub trait WriteOnlyBuffer: sealed::Sealed {
    type Item;

    /// Returns the length of the buffer _including_ any uninitialized memory.
    fn raw_buffer_len(&self) -> usize;

    /// Returns a mutable pointer to the buffer that is only able to be written to.
    ///
    /// # Safety
    ///
    /// This buffer must _never_ be read from as it can contain uninitialized bytes.
    unsafe fn as_write_only_ptr(&mut self) -> *mut Self::Item;

    #[inline(always)]
    /// Write a value to a specific index in the buffer.
    ///
    /// # Safety
    ///
    /// The `idx` must be within bounds of the buffer length.
    unsafe fn write_at(&mut self, idx: usize, value: Self::Item) {
        let ptr = self.as_write_only_ptr();
        ptr.add(idx).write(value)
    }
}

mod sealed {
    use core::mem::MaybeUninit;

    pub trait Sealed {}

    impl Sealed for &mut [f32] {}
    impl Sealed for &mut [f64] {}

    impl Sealed for &mut [i8] {}
    impl Sealed for &mut [i16] {}
    impl Sealed for &mut [i32] {}
    impl Sealed for &mut [i64] {}

    impl Sealed for &mut [u8] {}
    impl Sealed for &mut [u16] {}
    impl Sealed for &mut [u32] {}
    impl Sealed for &mut [u64] {}

    impl Sealed for &mut [MaybeUninit<f32>] {}
    impl Sealed for &mut [MaybeUninit<f64>] {}

    impl Sealed for &mut [MaybeUninit<i8>] {}
    impl Sealed for &mut [MaybeUninit<i16>] {}
    impl Sealed for &mut [MaybeUninit<i32>] {}
    impl Sealed for &mut [MaybeUninit<i64>] {}

    impl Sealed for &mut [MaybeUninit<u8>] {}
    impl Sealed for &mut [MaybeUninit<u16>] {}
    impl Sealed for &mut [MaybeUninit<u32>] {}
    impl Sealed for &mut [MaybeUninit<u64>] {}
}

macro_rules! add_slice_impl {
    ($t:ty, inner = $inner:ty) => {
        impl WriteOnlyBuffer for &mut [$t] {
            type Item = $inner;

            #[inline(always)]
            fn raw_buffer_len(&self) -> usize {
                self.len()
            }

            #[inline(always)]
            unsafe fn as_write_only_ptr(&mut self) -> *mut Self::Item {
                self.as_mut_ptr().cast()
            }
        }
    };
}

// Initialised impls
add_slice_impl!(f32, inner = f32);
add_slice_impl!(f64, inner = f64);

add_slice_impl!(i8, inner = i8);
add_slice_impl!(i16, inner = i16);
add_slice_impl!(i32, inner = i32);
add_slice_impl!(i64, inner = i64);

add_slice_impl!(u8, inner = u8);
add_slice_impl!(u16, inner = u16);
add_slice_impl!(u32, inner = u32);
add_slice_impl!(u64, inner = u64);

// Uninit impls
add_slice_impl!(MaybeUninit<f32>, inner = f32);
add_slice_impl!(MaybeUninit<f64>, inner = f64);

add_slice_impl!(MaybeUninit<i8>, inner = i8);
add_slice_impl!(MaybeUninit<i16>, inner = i16);
add_slice_impl!(MaybeUninit<i32>, inner = i32);
add_slice_impl!(MaybeUninit<i64>, inner = i64);

add_slice_impl!(MaybeUninit<u8>, inner = u8);
add_slice_impl!(MaybeUninit<u16>, inner = u16);
add_slice_impl!(MaybeUninit<u32>, inner = u32);
add_slice_impl!(MaybeUninit<u64>, inner = u64);
