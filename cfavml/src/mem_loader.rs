use std::fmt::Debug;

use crate::danger::{DenseLane, SimdRegister};

/// The stack scratch space used by the projecting buffer loader.
///
/// This is calculated by effectively taking the maximum number of elements
/// that could be loaded from the widest supported register in CFAVML, in
/// this case; AVX512.
const SCRATCH_SPACE_SIZE: usize = 64;

/// A buffer or value that can be turned into a [MemLoader].
///
/// NOTE: You are not supposed to implement this trait yourself.
pub trait IntoMemLoader<T> {
    /// The actual [MemLoader] produced by this implementation.
    type Loader: MemLoader<Value = T>;

    /// Consumes the value returning the configured [MemLoader]
    fn into_mem_loader(self) -> Self::Loader;

    /// Consumes the value returning the configured [MemLoader]
    ///
    /// # Panics
    ///
    /// Panics if the vector cannot be projected into the input size.
    fn into_projected_mem_loader(self, projected_len: usize) -> Self::Loader;
}

/// A trait that provides generic memory access and loading patterns for SIMD routines.
pub trait MemLoader {
    /// The inner value type within the loader.
    type Value: Copy;

    /// The true length of the buffer or value
    fn true_len(&self) -> usize;

    /// The length of the buffer the loader is potentially pretending to be.
    ///
    /// This is used in situations where the value is being broadcast to match a new size.
    fn projected_len(&self) -> usize;

    /// Performs an unsafe load of a dense lane from the [MemLoader] and advances
    //     /// the statemachine.
    ///
    /// # Safety
    ///
    /// This method has no concept of checking the remaining length of the loader,
    /// out of bounds access can easily happen if the routine does not track the current
    /// positions of buffers.
    unsafe fn load_dense<R: SimdRegister<Self::Value>>(
        &mut self,
    ) -> DenseLane<R::Register>;

    /// Performs an unsafe load of a single register from the [MemLoader] and advances
    //     /// the statemachine.
    ///
    /// # Safety
    ///
    /// This method has no concept of checking the remaining length of the loader,
    /// out of bounds access can easily happen if the routine does not track the current
    /// positions of buffers.
    unsafe fn load<R: SimdRegister<Self::Value>>(&mut self) -> R::Register;

    /// Performs an unsafe load of a single value from the [MemLoader] and advances
    /// the statemachine.
    ///
    /// # Safety
    ///
    /// This method has no concept of checking the remaining length of the loader,
    /// out of bounds access can easily happen if the routine does not track the current
    /// positions of buffers.
    unsafe fn read(&mut self) -> Self::Value;
}

impl<'a, B, T> IntoMemLoader<T> for &'a B
where
    T: Copy,
    B: AsRef<[T]> + ?Sized,
{
    type Loader = PtrBufferLoader<T>;

    fn into_projected_mem_loader(self, projected_len: usize) -> Self::Loader {
        let slice = self.as_ref();

        assert_eq!(
            slice.len(),
            projected_len,
            "Input slice does not match target output length, \
            by default slices cannot be projected to a new size. \
            You can enable projection to new sizes by wrapping your value in \
            a `Projected<T>` wrapper."
        );

        self.into_mem_loader()
    }

    fn into_mem_loader(self) -> Self::Loader {
        let slice = self.as_ref();
        PtrBufferLoader {
            data: slice.as_ptr(),
            data_len: slice.len(),
            data_cursor: 0,
        }
    }
}

/// A wrapper that enables extended projection of the input buffer
/// to a new shape / size.
///
/// Please be aware that this type only supports projecting default
/// implementations of _slices_ provided by this library, it does
/// not support handling custom MemLoader implementations.
///
/// ## Projection Rules
///
/// A buffer can be projected to a new size providing
/// the _new_ size is a multiple of the _old_ size.
///
/// For example, we can project any of the following:
///
/// - `size:40 -> size:80`
/// - `size:4 -> size:16`
/// - `size:1 -> size:73`
///
/// But we cannot project:
///
/// - `size:3` -> `size:4`
/// - `size:2` -> `size:9`
///
/// ## Projection Behaviour
///
/// This projection system is _not_ like numpy broadcasting or other ndarray-like
/// broadcasting it is only aware of the _length_ of the buffer, not whether it is a
/// matrix or a type which has a shape.
///
/// Because of this, this routine may behave differently to what you expect, allowing
/// say the projection of a matrix (represented as a slice) of shape `(4, 4)` being
/// broadcast to shape `(8, 4)` because _technically_ there is no difference in
/// array size of shapes `(2, 4, 4)` and `(8, 4)` it is simply a multiple of `16 (4, 4)`.
///
///
pub struct Projected<T>(pub T);

impl<'a, B, T> IntoMemLoader<T> for Projected<&'a B>
where
    T: Copy + Default + Debug,
    B: AsRef<[T]> + ?Sized,
{
    type Loader = ProjectedPtrBufferLoader<T>;

    fn into_projected_mem_loader(self, projected_len: usize) -> Self::Loader {
        let slice = self.0.as_ref();

        assert_eq!(
            projected_len % slice.len(),
            0,
            "Cannot project slice into size {projected_len}, because it is not a multiple of {}",
            slice.len(),
        );

        ProjectedPtrBufferLoader {
            data: slice.as_ptr(),
            data_len: slice.len(),
            data_cursor: 0,
            projected_len,
        }
    }

    fn into_mem_loader(self) -> Self::Loader {
        let slice = self.0.as_ref();
        ProjectedPtrBufferLoader {
            data: slice.as_ptr(),
            data_len: slice.len(),
            data_cursor: 0,
            projected_len: slice.len(),
        }
    }
}

macro_rules! impl_scalar_buffer_loader {
    ($t:ty) => {
        impl IntoMemLoader<$t> for $t {
            type Loader = ScalarBufferLoader<$t>;

            fn into_projected_mem_loader(self, projected_len: usize) -> Self::Loader {
                ScalarBufferLoader {
                    data: self,
                    projected_len,
                }
            }

            fn into_mem_loader(self) -> Self::Loader {
                ScalarBufferLoader {
                    data: self,
                    projected_len: 1,
                }
            }
        }
    };
}

impl_scalar_buffer_loader!(f32);
impl_scalar_buffer_loader!(f64);
impl_scalar_buffer_loader!(i8);
impl_scalar_buffer_loader!(i16);
impl_scalar_buffer_loader!(i32);
impl_scalar_buffer_loader!(i64);
impl_scalar_buffer_loader!(u8);
impl_scalar_buffer_loader!(u16);
impl_scalar_buffer_loader!(u32);
impl_scalar_buffer_loader!(u64);

/// A [MemLoader] implementation that reads from a contiguous buffer represented
/// as a data pointer which can be projected to a size greater than its own.
pub struct PtrBufferLoader<T> {
    data: *const T,
    data_len: usize,

    // Generator state machine
    data_cursor: usize,
}

impl<T: Copy> MemLoader for PtrBufferLoader<T> {
    type Value = T;

    #[inline(always)]
    fn true_len(&self) -> usize {
        self.data_len
    }

    #[inline(always)]
    fn projected_len(&self) -> usize {
        self.data_len
    }

    #[inline(always)]
    unsafe fn load_dense<R: SimdRegister<Self::Value>>(
        &mut self,
    ) -> DenseLane<R::Register> {
        let dense = R::load_dense(self.data.add(self.data_cursor));
        self.data_cursor += R::elements_per_dense();
        dense
    }

    #[inline(always)]
    unsafe fn load<R: SimdRegister<Self::Value>>(&mut self) -> R::Register {
        let dense = R::load(self.data.add(self.data_cursor));
        self.data_cursor += R::elements_per_lane();
        dense
    }

    #[inline(always)]
    unsafe fn read(&mut self) -> Self::Value {
        let value = self.data.add(self.data_cursor).read();
        self.data_cursor += 1;
        value
    }
}

/// A [MemLoader] implementation that reads from a contiguous buffer represented
/// as a data pointer which can be projected to a size greater than its own.
pub struct ProjectedPtrBufferLoader<T> {
    data: *const T,
    data_len: usize,

    // Generator state machine
    data_cursor: usize,
    projected_len: usize,
}

impl<T: Copy + Debug> ProjectedPtrBufferLoader<T> {
    fn can_load_full_dense_lane<R: SimdRegister<T>>(&self) -> bool {
        self.data_cursor + R::elements_per_dense() <= self.data_len
    }

    fn can_load_full_lane<R: SimdRegister<T>>(&self) -> bool {
        self.data_cursor + R::elements_per_lane() <= self.data_len
    }

    fn advance_cursor(&mut self, by: usize) {
        self.data_cursor = (self.data_cursor + by) % self.data_len;
    }
}

impl<T: Copy + Default + Debug> MemLoader for ProjectedPtrBufferLoader<T> {
    type Value = T;

    #[inline(always)]
    fn true_len(&self) -> usize {
        self.data_len
    }

    #[inline(always)]
    fn projected_len(&self) -> usize {
        self.projected_len
    }

    #[inline(always)]
    unsafe fn load_dense<R: SimdRegister<Self::Value>>(
        &mut self,
    ) -> DenseLane<R::Register> {
        if self.can_load_full_dense_lane::<R>() {
            let dense = R::load_dense(self.data.add(self.data_cursor));
            self.advance_cursor(R::elements_per_dense());
            return dense;
        }

        DenseLane {
            a: self.load::<R>(),
            b: self.load::<R>(),
            c: self.load::<R>(),
            d: self.load::<R>(),
            e: self.load::<R>(),
            f: self.load::<R>(),
            g: self.load::<R>(),
            h: self.load::<R>(),
        }
    }

    #[inline(always)]
    unsafe fn load<R: SimdRegister<Self::Value>>(&mut self) -> R::Register {
        if self.can_load_full_lane::<R>() {
            let dense = R::load(self.data.add(self.data_cursor));
            self.advance_cursor(R::elements_per_lane());
            return dense;
        }

        let mut temp_buffer = [T::default(); SCRATCH_SPACE_SIZE];

        // elements_per_lane != SCRATCH_SPACE_SIZE, this is cleaner than an iter chain.
        #[allow(clippy::needless_range_loop)]
        for i in 0..R::elements_per_lane() {
            temp_buffer[i] = self.read();
        }
        dbg!(&temp_buffer);

        R::load(temp_buffer.as_ptr())
    }

    #[inline(always)]
    unsafe fn read(&mut self) -> Self::Value {
        let value = self.data.add(self.data_cursor).read();
        self.advance_cursor(1);
        value
    }
}

/// A [MemLoader] implementation that holds a single value that has been broadcast
/// to a desired size.
pub struct ScalarBufferLoader<T> {
    data: T,
    projected_len: usize,
}

impl<T: Copy> MemLoader for ScalarBufferLoader<T> {
    type Value = T;

    #[inline(always)]
    fn true_len(&self) -> usize {
        1
    }

    #[inline(always)]
    fn projected_len(&self) -> usize {
        self.projected_len
    }

    #[inline(always)]
    unsafe fn load_dense<R: SimdRegister<Self::Value>>(
        &mut self,
    ) -> DenseLane<R::Register> {
        R::filled_dense(self.data)
    }

    #[inline(always)]
    unsafe fn load<R: SimdRegister<Self::Value>>(&mut self) -> R::Register {
        R::filled(self.data)
    }

    #[inline(always)]
    unsafe fn read(&mut self) -> Self::Value {
        self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::danger::Fallback;

    #[test]
    fn test_value_broadcast_loader() {
        let mut loader = f32::into_mem_loader(1.0);
        assert_eq!(loader.projected_len(), 1);
        let a = unsafe { loader.read() };
        assert_eq!(a, 1.0);

        let mut loader = f32::into_projected_mem_loader(1.0, 10);
        assert_eq!(loader.projected_len(), 10);
        for _ in 0..10 {
            let a = unsafe { loader.read() };
            assert_eq!(a, 1.0);
        }
    }

    #[allow(clippy::needless_range_loop)]
    #[test]
    fn test_buffer_basic_loader() {
        let sample = [1.0, 2.0, 3.0];
        let mut loader = (&sample).into_mem_loader();
        assert_eq!(loader.projected_len(), 3);
        for i in 0..3 {
            assert_eq!(unsafe { loader.read() }, sample[i]);
        }
    }

    #[test]
    #[should_panic]
    fn test_buffer_basic_loader_projection_panic() {
        let sample = [1.0, 2.0, 3.0];
        let _loader = (&sample).into_projected_mem_loader(10);
    }

    #[test]
    #[should_panic]
    fn test_buffer_projection_creation_panic() {
        let sample = [1.0, 2.0];
        let projected = Projected(&sample);
        let _loader = projected.into_projected_mem_loader(5);
    }

    #[test]
    fn test_buffer_projection_basic_read() {
        let sample = [1.0, 2.0];
        let projected = Projected(&sample);
        let mut loader = projected.into_projected_mem_loader(4);
        assert_eq!(loader.projected_len(), 4);

        unsafe {
            assert_eq!(loader.read(), 1.0);
            assert_eq!(loader.read(), 2.0);
            assert_eq!(loader.read(), 1.0);
            assert_eq!(loader.read(), 2.0);
        }
    }

    #[test]
    fn test_buffer_projection_fallback_dense_load() {
        let sample = [1.0, 2.0];
        let projected = Projected(&sample);
        let mut loader = projected.into_projected_mem_loader(4);
        assert_eq!(loader.projected_len(), 4);

        unsafe {
            let dense = loader.load_dense::<Fallback>();
            assert_eq!(dense.a, 1.0);
            assert_eq!(dense.b, 2.0);
            assert_eq!(dense.c, 1.0);
            assert_eq!(dense.d, 2.0);
            assert_eq!(dense.e, 1.0);
            assert_eq!(dense.f, 2.0);
            assert_eq!(dense.g, 1.0);
            assert_eq!(dense.h, 2.0);
        }
    }

    #[test]
    fn test_buffer_projection_fallback_load() {
        let sample = [1.0, 2.0];
        let projected = Projected(&sample);
        let mut loader = projected.into_projected_mem_loader(4);
        assert_eq!(loader.projected_len(), 4);

        unsafe {
            let reg = loader.load::<Fallback>();
            assert_eq!(reg, 1.0);
            let reg = loader.load::<Fallback>();
            assert_eq!(reg, 2.0);
            let reg = loader.load::<Fallback>();
            assert_eq!(reg, 1.0);
            let reg = loader.load::<Fallback>();
            assert_eq!(reg, 2.0);
        }
    }

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    ))]
    #[test]
    fn test_buffer_projection_avx2_dense_load() {
        let sample = [1.0f32, 2.0f32];
        let projected = Projected(&sample);
        let mut loader = projected.into_projected_mem_loader(4);
        assert_eq!(loader.projected_len(), 4);

        #[allow(clippy::missing_transmute_annotations)]
        unsafe {
            let dense = loader.load_dense::<crate::danger::Avx2>();
            assert_eq!(
                core::mem::transmute::<_, [f32; 8]>(dense.a),
                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
            );
            assert_eq!(
                core::mem::transmute::<_, [f32; 8]>(dense.b),
                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
            );
            assert_eq!(
                core::mem::transmute::<_, [f32; 8]>(dense.c),
                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
            );
            assert_eq!(
                core::mem::transmute::<_, [f32; 8]>(dense.d),
                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
            );
            assert_eq!(
                core::mem::transmute::<_, [f32; 8]>(dense.e),
                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
            );
            assert_eq!(
                core::mem::transmute::<_, [f32; 8]>(dense.f),
                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
            );
            assert_eq!(
                core::mem::transmute::<_, [f32; 8]>(dense.g),
                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
            );
            assert_eq!(
                core::mem::transmute::<_, [f32; 8]>(dense.h),
                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
            );
        }

        let sample = [1.0f32, 2.0f32, 3.0f32];
        let projected = Projected(&sample);
        let mut loader = projected.into_projected_mem_loader(9);
        assert_eq!(loader.projected_len(), 9);

        #[allow(clippy::missing_transmute_annotations)]
        unsafe {
            let dense = loader.load_dense::<crate::danger::Avx2>();
            assert_eq!(
                core::mem::transmute::<_, [f32; 8]>(dense.a),
                [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]
            );
            assert_eq!(
                core::mem::transmute::<_, [f32; 8]>(dense.b),
                [3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]
            );
            assert_eq!(
                core::mem::transmute::<_, [f32; 8]>(dense.c),
                [2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
            );
            assert_eq!(
                core::mem::transmute::<_, [f32; 8]>(dense.d),
                [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]
            );
            assert_eq!(
                core::mem::transmute::<_, [f32; 8]>(dense.e),
                [3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]
            );
            assert_eq!(
                core::mem::transmute::<_, [f32; 8]>(dense.f),
                [2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
            );
            assert_eq!(
                core::mem::transmute::<_, [f32; 8]>(dense.g),
                [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]
            );
            assert_eq!(
                core::mem::transmute::<_, [f32; 8]>(dense.h),
                [3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]
            );
        }
    }

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    ))]
    #[test]
    fn test_buffer_projection_avx2_load() {
        let sample = [1.0f32, 2.0f32];
        let projected = Projected(&sample);
        let mut loader = projected.into_projected_mem_loader(4);
        assert_eq!(loader.projected_len(), 4);

        #[allow(clippy::missing_transmute_annotations)]
        unsafe {
            let reg = loader.load::<crate::danger::Avx2>();
            assert_eq!(
                core::mem::transmute::<_, [f32; 8]>(reg),
                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
            );
            let reg = loader.load::<crate::danger::Avx2>();
            assert_eq!(
                core::mem::transmute::<_, [f32; 8]>(reg),
                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
            );
            let reg = loader.load::<crate::danger::Avx2>();
            assert_eq!(
                core::mem::transmute::<_, [f32; 8]>(reg),
                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
            );
            let reg = loader.load::<crate::danger::Avx2>();
            assert_eq!(
                core::mem::transmute::<_, [f32; 8]>(reg),
                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
            );
        }

        let sample = [1.0f32, 2.0f32, 3.0f32];
        let projected = Projected(&sample);
        let mut loader = projected.into_projected_mem_loader(9);
        assert_eq!(loader.projected_len(), 9);

        #[allow(clippy::missing_transmute_annotations)]
        unsafe {
            let reg = loader.load::<crate::danger::Avx2>();
            assert_eq!(
                core::mem::transmute::<_, [f32; 8]>(reg),
                [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]
            );
            let reg = loader.load::<crate::danger::Avx2>();
            assert_eq!(
                core::mem::transmute::<_, [f32; 8]>(reg),
                [3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]
            );
            let reg = loader.load::<crate::danger::Avx2>();
            assert_eq!(
                core::mem::transmute::<_, [f32; 8]>(reg),
                [2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
            );
        }
    }

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx512f",
        feature = "nightly"
    ))]
    #[test]
    fn test_buffer_projection_avx512_dense_load() {
        let sample = [1.0f64, 2.0f64];
        let projected = Projected(&sample);
        let mut loader = projected.into_projected_mem_loader(4);
        assert_eq!(loader.projected_len(), 4);

        #[allow(clippy::missing_transmute_annotations)]
        unsafe {
            let dense = loader.load_dense::<crate::danger::Avx512>();
            assert_eq!(
                core::mem::transmute::<_, [f64; 8]>(dense.a),
                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
            );
            assert_eq!(
                core::mem::transmute::<_, [f64; 8]>(dense.b),
                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
            );
            assert_eq!(
                core::mem::transmute::<_, [f64; 8]>(dense.c),
                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
            );
            assert_eq!(
                core::mem::transmute::<_, [f64; 8]>(dense.d),
                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
            );
            assert_eq!(
                core::mem::transmute::<_, [f64; 8]>(dense.e),
                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
            );
            assert_eq!(
                core::mem::transmute::<_, [f64; 8]>(dense.f),
                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
            );
            assert_eq!(
                core::mem::transmute::<_, [f64; 8]>(dense.g),
                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
            );
            assert_eq!(
                core::mem::transmute::<_, [f64; 8]>(dense.h),
                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
            );
        }

        let sample = [1.0f64, 2.0f64, 3.0f64];
        let projected = Projected(&sample);
        let mut loader = projected.into_projected_mem_loader(9);
        assert_eq!(loader.projected_len(), 9);

        #[allow(clippy::missing_transmute_annotations)]
        unsafe {
            let dense = loader.load_dense::<crate::danger::Avx512>();
            assert_eq!(
                core::mem::transmute::<_, [f64; 8]>(dense.a),
                [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]
            );
            assert_eq!(
                core::mem::transmute::<_, [f64; 8]>(dense.b),
                [3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]
            );
            assert_eq!(
                core::mem::transmute::<_, [f64; 8]>(dense.c),
                [2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
            );
            assert_eq!(
                core::mem::transmute::<_, [f64; 8]>(dense.d),
                [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]
            );
            assert_eq!(
                core::mem::transmute::<_, [f64; 8]>(dense.e),
                [3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]
            );
            assert_eq!(
                core::mem::transmute::<_, [f64; 8]>(dense.f),
                [2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
            );
            assert_eq!(
                core::mem::transmute::<_, [f64; 8]>(dense.g),
                [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]
            );
            assert_eq!(
                core::mem::transmute::<_, [f64; 8]>(dense.h),
                [3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]
            );
        }
    }

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx512f",
        feature = "nightly",
    ))]
    #[test]
    fn test_buffer_projection_avx512_load() {
        let sample = [1.0f64, 2.0f64];
        let projected = Projected(&sample);
        let mut loader = projected.into_projected_mem_loader(4);
        assert_eq!(loader.projected_len(), 4);

        #[allow(clippy::missing_transmute_annotations)]
        unsafe {
            let reg = loader.load::<crate::danger::Avx512>();
            assert_eq!(
                core::mem::transmute::<_, [f64; 8]>(reg),
                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
            );
            let reg = loader.load::<crate::danger::Avx512>();
            assert_eq!(
                core::mem::transmute::<_, [f64; 8]>(reg),
                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
            );
            let reg = loader.load::<crate::danger::Avx512>();
            assert_eq!(
                core::mem::transmute::<_, [f64; 8]>(reg),
                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
            );
            let reg = loader.load::<crate::danger::Avx512>();
            assert_eq!(
                core::mem::transmute::<_, [f64; 8]>(reg),
                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
            );
        }

        let sample = [1.0f64, 2.0f64, 3.0f64];
        let projected = Projected(&sample);
        let mut loader = projected.into_projected_mem_loader(9);
        assert_eq!(loader.projected_len(), 9);

        #[allow(clippy::missing_transmute_annotations)]
        unsafe {
            let reg = loader.load::<crate::danger::Avx512>();
            assert_eq!(
                core::mem::transmute::<_, [f64; 8]>(reg),
                [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]
            );
            let reg = loader.load::<crate::danger::Avx512>();
            assert_eq!(
                core::mem::transmute::<_, [f64; 8]>(reg),
                [3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]
            );
            let reg = loader.load::<crate::danger::Avx512>();
            assert_eq!(
                core::mem::transmute::<_, [f64; 8]>(reg),
                [2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
            );
        }
    }
}
