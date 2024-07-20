use cfavml::danger::{DenseLane, SimdRegister};
use cfavml::math::{AutoMath, Math};


/// A generic matrix micro kernel.
///
/// The reason bits are being used here is because this kernel is generic over
/// the actual type being used, so for `f32` this becomes a 8x8 kernel, `f64` becomes
/// `4x4` etc... Internally these all become operations over a [SimdRegister] with
/// a 8x dense lane.
pub struct GenericMatrixMicroKernel<T, R, M = AutoMath>
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
    a_buffer: Vec<()>,
    /// Our column components, primarily targeting to be cached in L1, but this is largely
    /// up to the CPU to do so, this buffer is loaded in _rows_ rather than columns.
    b_buffer: Vec<()>,
    /// Our matmul result for the micro kernel, these are kept within registers.
    matrix_result: DenseLane<R::Register>,
}

impl<T, R, M> GenericMatrixMicroKernel<T, R, M>
where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
{
    /// Creates a new micro kernel with an allocation large enough
    /// to handle the 256bit x 8 operation.
    pub fn allocate(dims: usize) -> Self {

    }
}