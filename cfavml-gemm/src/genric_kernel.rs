use std::marker::PhantomData;
use cfavml::danger::{DenseLane, SimdRegister};
use cfavml::math::{AutoMath, Math};
use cfavml_utils::aligned_buffer::AlignedBuffer;

/// A generic matrix micro kernel.
///
/// The reason bits are being used here is because this kernel is generic over
/// the actual type being used, so for `f32` this becomes a 8x8 kernel, `f64` becomes
/// `4x4` etc... Internally these all become operations over a [SimdRegister] with
/// a 8x dense lane.
pub(crate) struct GenericMatrixMicroKernel<T, R, M = AutoMath>
where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
{
    /// THe number of dimensions of the individual vectors of the matrix.
    dims: usize,
    /// Our row components, primarily targeting to be cached in L2, but this is largely
    /// up to the CPU to do so, each element is loaded individually and broadcast
    /// across registers.
    a_buffer: AlignedBuffer<T>,
    /// Our column components, primarily targeting to be cached in L1, but this is largely
    /// up to the CPU to do so, this buffer is loaded in _rows_ rather than columns.
    b_buffer: AlignedBuffer<T>,
    /// Our matmul result for the micro kernel, these are kept within registers.
    matrix_result: DenseLane<R::Register>,
    math: PhantomData<M>,
}

impl<T, R, M> GenericMatrixMicroKernel<T, R, M>
where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
{
    /// Creates a new micro kernel with an allocation large enough
    /// to handle the 256bit x 8 operation.
    pub(crate) fn allocate(dims: usize) -> Self {
        // We select `num_elements_per_lane * 8` rows+cols to compute
        // the dot product with, this means on f32 this is `8 * 8 == 64` elements
        // amounting to `8 * dims` elements loaded from buffers `a` and `b`.
        let num_elements_per_lane = R::elements_per_lane();
        let buffer_size = dims * num_elements_per_lane;

        // SAFETY:
        // Internally this is only ever going to be used with integer primitives
        // which do not impose any additional alignment issues.
        let a_buffer = unsafe { AlignedBuffer::zeroed(buffer_size) };
        let b_buffer = unsafe { AlignedBuffer::zeroed(buffer_size) };

        Self {
            dims,
            a_buffer,
            b_buffer,
            matrix_result: unsafe { R::zeroed_dense() },
            math: PhantomData,
        }
    }

    /// Computes the dot product of the given matrix of `a` and `b` writing the result to `result`
    /// using the configured micro-kernel.
    pub(crate) fn dot_matrix(
        &mut self,
        a: &[T],
        b: &[T],
        result: &mut [T],
    ) {

    }

    /// Executes the micro-kernel using the configured buffers.
    unsafe fn execute_kernel(&mut self) {
        let a_ptr = self.a_buffer.as_ptr();
        let b_ptr = self.b_buffer.as_ptr();

        // So future me:
        // Here is the plan of the first generation because I can't remember how the BLIS papers
        // did it...
        //
        // - We do an initial naive test loading a and broadcasting it across registers.
        // - Then we go over the _full_ row in B and do the dot product intermediate result for
        //   that row.
        // - Then we advance `a` by 1, load, broadcast, repeat.
        // - No idea how this is gonna go, maybe it is worth transposing `a` to be column-major?
        //   * Reason I say this is because it is a bit of a waste to go through `b` completely
        //     for just one element of `a`, since that ends up being `dims` iterations per element
        //     of `a` and loading the memory each time!
        //   * Maybe transposing to column-major is too expensive here though.

        let a_broadcast = R::filled_dense(a_ptr.read());

        let b_row_chunks = R::load_dense(b_ptr);

        // Store intermediary back, since we are calculating the first step of the first row.
        self.matrix_result.a = R::fmadd_dense(a_broadcast, b_row_chunks, self.matrix_result.a);

        self.matrix_result = R::zeroed_dense();
    }
}
