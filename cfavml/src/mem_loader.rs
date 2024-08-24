use crate::danger::{DenseLane, SimdRegister};

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
        assert_eq!(
            projected_len,
            self.as_ref().len(),
            "Buffer cannot be projected outside of its existing dimensions currently",
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
/// as a data pointer.
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
}
