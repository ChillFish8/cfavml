use std::any::TypeId;
use std::mem;

use cfavml::danger::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod impl_avx2;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::impl_avx2::*;

/// Transpose a given matrix, writing the result to the given output buffer.
pub fn transpose_matrix<T>(width: usize, height: usize, data: &[T], result: &mut [T])
where
    T: Copy + 'static,
{
    assert_eq!(data.len(), width * height, "Input data shape missmatch");
    assert_eq!(
        data.len(),
        result.len(),
        "Output buffer does not match input data"
    );

    if width == 0 || height == 0 {
        return;
    }

    if width == 1 || height == 1 {
        result.copy_from_slice(data);
        return;
    }

    if TypeId::of::<T>() == TypeId::of::<f32>()
        || TypeId::of::<T>() == TypeId::of::<u32>()
    {
        let data = unsafe { mem::transmute::<&[T], &[f32]>(data) };
        let result = unsafe { mem::transmute::<&mut [T], &mut [f32]>(result) };

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if is_x86_feature_detected!("avx2") {
                return f32_xany_avx2_nofma_transpose(width, height, data, result)
            }
        }
    } else if TypeId::of::<T>() == TypeId::of::<f64>()
        || TypeId::of::<T>() == TypeId::of::<u64>()
    {
        let data = unsafe { mem::transmute::<&[T], &[f64]>(data) };
        let result = unsafe { mem::transmute::<&mut [T], &mut [f64]>(result) };

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if is_x86_feature_detected!("avx2") {
                return f64_xany_avx2_nofma_transpose(width, height, data, result)
            }
        }
    }

    // Any remaining cases falls back to a naive solution.
    let mut j = 0;
    while j < height {
        let mut i = 0;
        while i < width {
            unsafe {
                *result.get_unchecked_mut(i * height + j) = *data.get_unchecked(j * width + i);
            }

            i += 1;
        }

        j += 1;
    }
}

/// Transpose a full width x height matrix.
unsafe fn generic_transpose<T, R>(
    width: usize,
    height: usize,
    data: &[T],
    result: &mut [T],
) where
    T: Copy,
    R: SimdRegister<T> + TransposeMatrix<T>,
{
    assert_eq!(data.len(), width * height, "Input data shape missmatch");
    assert_eq!(
        data.len(),
        result.len(),
        "Output buffer does not match input data"
    );

    let data_ptr = data.as_ptr();
    let result_ptr = result.as_mut_ptr();

    let sub_matrix_block_size = R::elements_per_lane() * 2;
    let matrix_offset_step = R::elements_per_lane();

    let width_remainder = width % sub_matrix_block_size;
    let height_remainder = height % sub_matrix_block_size;

    let mut j = 0;
    while j < (height - height_remainder) {
        let mut i = 0;
        while i < (width - width_remainder) {
            // By performing a 2Nx2N matrix transposition made up of
            // 8x8 sub-operations we maintain cache locality as best we can
            // Increasing the size to 4Nx4N did not appear to provide and additional
            // benefit and was overall a net less according to benchmarks.

            // top-left
            let l1 = R::load_matrix(i + j * width, width, data_ptr);
            let l1_transpose = R::transpose_register_matrix(l1);
            R::write_matrix(j + i * height, height, l1_transpose, result_ptr);

            // bottom-left
            let l1 =
                R::load_matrix(i + (j + matrix_offset_step) * width, width, data_ptr);
            let l1_transpose = R::transpose_register_matrix(l1);
            R::write_matrix(
                (j + matrix_offset_step) + i * height,
                height,
                l1_transpose,
                result_ptr,
            );

            // top-right
            let l1 =
                R::load_matrix((i + matrix_offset_step) + j * width, width, data_ptr);
            let l1_transpose = R::transpose_register_matrix(l1);
            R::write_matrix(
                j + (i + matrix_offset_step) * height,
                height,
                l1_transpose,
                result_ptr,
            );

            // bottom-right
            let l1 = R::load_matrix(
                (i + matrix_offset_step) + (j + matrix_offset_step) * width,
                width,
                data_ptr,
            );
            let l1_transpose = R::transpose_register_matrix(l1);
            R::write_matrix(
                (j + matrix_offset_step) + (i + matrix_offset_step) * height,
                height,
                l1_transpose,
                result_ptr,
            );

            i += sub_matrix_block_size;
        }

        j += sub_matrix_block_size;
    }

    // Handles the tail of each row that does not fit within 8 wide blocks.
    let mut j_remainder = 0;
    while j_remainder < (height - height_remainder) {
        let mut i = width - width_remainder;
        while i < width {
            *result.get_unchecked_mut(i * height + j_remainder) =
                *data.get_unchecked(j_remainder * width + i);

            i += 1;
        }

        j_remainder += 1;
    }

    // Handles the tail of each column that does not fit within 8 wide blocks.
    while j < height {
        let mut i = 0;
        while i < width {
            *result.get_unchecked_mut(i * height + j) =
                *data.get_unchecked(j * width + i);

            i += 1;
        }

        j += 1;
    }
}

/// Generic matrix transposition.
///
/// The implementation of the [TransposeMatrix] operations
/// process blocks in 4x4 sub-matrices, assuming sub-matrices will be
/// made up of 4 [Self::RegisterMatrix] types in order to maximise
/// cache locality.
trait TransposeMatrix<T>: SimdRegister<T>
where
    T: Copy,
{
    /// The type representing `NxN` matrix within SIMD registers.
    type RegisterMatrix;

    /// Load a new `NxN` matrix into registers.
    unsafe fn load_matrix(
        offset: usize,
        width: usize,
        data_ptr: *const T,
    ) -> Self::RegisterMatrix;

    /// Store a `NxN` matrix in the `result_ptr` buffer.
    unsafe fn write_matrix(
        offset: usize,
        height: usize,
        matrix: Self::RegisterMatrix,
        result_ptr: *mut T,
    );

    /// Transpose the [Self::RegisterMatrix] instance returning the transposed data.
    unsafe fn transpose_register_matrix(
        matrix: Self::RegisterMatrix,
    ) -> Self::RegisterMatrix;
}
