use core::arch::x86_64::*;

use crate::danger::{offsets_avx2_ps, rollup_x8_ps, sum_avx2_ps, CHUNK_0, CHUNK_1};

#[target_feature(enable = "avx2")]
#[inline]
/// Sums all elements of the vector.
///
/// ```py
/// D: int
/// total: f32
/// x: [f32; D]
///
/// for i in 0..D:
///     total = total + x[i]
/// ```
///
/// # Safety
///
/// `DIMS` **MUST** be a multiple of `64`, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// This method assumes AVX2 instructions are available, if this method is executed
/// on non-AVX2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xconst_avx2_nofma_sum_horizontal<const DIMS: usize>(x: &[f32]) -> f32 {
    debug_assert_eq!(DIMS % 64, 0, "DIMS must be a multiple of 64");
    debug_assert_eq!(x.len(), DIMS);

    let x = x.as_ptr();

    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut acc4 = _mm256_setzero_ps();
    let mut acc5 = _mm256_setzero_ps();
    let mut acc6 = _mm256_setzero_ps();
    let mut acc7 = _mm256_setzero_ps();
    let mut acc8 = _mm256_setzero_ps();

    let mut i = 0;
    while i < DIMS {
        sum_x64_block(
            x.add(i),
            &mut acc1,
            &mut acc2,
            &mut acc3,
            &mut acc4,
            &mut acc5,
            &mut acc6,
            &mut acc7,
            &mut acc8,
        );

        i += 64;
    }

    let acc = rollup_x8_ps(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8);
    sum_avx2_ps(acc)
}

#[target_feature(enable = "avx2")]
#[inline]
/// Sums all elements of the vector.
///
/// ```py
/// D: int
/// total: f32
/// x: [f32; D]
///
/// for i in 0..D:
///     total = total + x[i]
/// ```
///
/// # Safety
///
/// This method assumes AVX2 instructions are available, if this method is executed
/// on non-AVX2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xany_avx2_nofma_sum_horizontal(x: &[f32]) -> f32 {
    let len = x.len();
    let offset_from = len % 64;

    let x_ptr = x.as_ptr();
    let mut extra = 0.0;

    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut acc4 = _mm256_setzero_ps();
    let mut acc5 = _mm256_setzero_ps();
    let mut acc6 = _mm256_setzero_ps();
    let mut acc7 = _mm256_setzero_ps();
    let mut acc8 = _mm256_setzero_ps();

    let mut i = 0;
    while i < (len - offset_from) {
        sum_x64_block(
            x_ptr.add(i),
            &mut acc1,
            &mut acc2,
            &mut acc3,
            &mut acc4,
            &mut acc5,
            &mut acc6,
            &mut acc7,
            &mut acc8,
        );

        i += 64;
    }

    if offset_from != 0 {
        let tail = offset_from % 8;

        while i < (len - tail) {
            let x = _mm256_loadu_ps(x_ptr.add(i));
            acc1 = _mm256_add_ps(acc1, x);

            i += 8;
        }

        while i < len {
            let x = *x.get_unchecked(i);
            extra += x;

            i += 1;
        }
    }

    let acc = rollup_x8_ps(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8);
    extra + sum_avx2_ps(acc)
}

#[target_feature(enable = "avx2")]
#[inline]
/// Vertical sum of the given matrix returning the individual sums.
///
/// ```py
/// DIMS: int
/// total: [f32; DIMS]
/// matrix: [[f32; DIMS]; N]
///
/// for i in 0..N:
///     for j in 0..DIMS:
///         total[j] += matrix[i, j]   
/// ```
///
/// # Safety
///
/// `DIMS` **MUST** be a multiple of `64`, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// The `matrix` must be a multiple of DIMS to produce N vectors within the matrix,
/// and `output` must be equal to DIMS.
///
/// This method assumes AVX2 instructions are available, if this method is executed
/// on non-AVX2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xconst_avx2_nofma_sum_vertical<const DIMS: usize>(
    matrix: &[f32],
    output: &mut [f32],
) {
    debug_assert_eq!(DIMS % 64, 0, "DIMS must be a multiple of 64");
    debug_assert_eq!(
        matrix.len() % DIMS,
        0,
        "Matrix shape must be `[[f32; DIMS]; N]`"
    );
    debug_assert_eq!(output.len(), DIMS, "Output buffer must be DIMS in size");

    let matrix_len = matrix.len();
    let matrix_ptr = matrix.as_ptr();
    let results_ptr = output.as_mut_ptr();

    let mut i = 0;
    while i < DIMS {
        sum_vertical_component(i, matrix_ptr, matrix_len, results_ptr, DIMS);

        i += 64;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
/// Vertical sum of the given matrix returning the individual sums.
///
/// ```py
/// D: int
/// total: [f32; D]
/// matrix: [[f32; D]; N]
///
/// for i in 0..N:
///     for j in 0..D:
///         total[j] += matrix[i, j]   
/// ```
///
/// # Safety
///
/// All vectors within the matrix **MUST** be the same length.
///
/// This method assumes AVX2 instructions are available, if this method is executed
/// on non-AVX2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xany_avx2_nofma_sum_vertical(matrix: &[f32], output: &mut [f32]) {
    let dims = output.len();
    let offset_from = dims % 64;

    debug_assert_eq!(
        matrix.len() % dims,
        0,
        "Matrix must have number of elements that is a multiple of dims",
    );

    let matrix_len = matrix.len();
    let matrix_ptr = matrix.as_ptr();

    let results_ptr = output.as_mut_ptr();

    let mut i = 0;
    while i < (dims - offset_from) {
        sum_vertical_component(i, matrix_ptr, matrix_len, results_ptr, dims);

        i += 64;
    }

    if offset_from != 0 {
        let tail = offset_from % 8;

        while i < (dims - tail) {
            let mut acc = _mm256_setzero_ps();

            let mut j = 0;
            while j < matrix_len {
                let x = _mm256_loadu_ps(matrix_ptr.add(j + i));
                acc = _mm256_add_ps(acc, x);

                j += dims;
            }

            _mm256_storeu_ps(results_ptr.add(i), acc);

            i += 8;
        }

        while i < dims {
            let mut j = 0;
            while j < matrix_len {
                *output.get_unchecked_mut(i) += *matrix.get_unchecked(j + i);

                j += dims;
            }

            i += 1;
        }
    }
}

#[inline(always)]
unsafe fn sum_vertical_component(
    i: usize,
    matrix_ptr: *const f32,
    matrix_len: usize,
    results_ptr: *mut f32,
    dims: usize,
) {
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut acc4 = _mm256_setzero_ps();
    let mut acc5 = _mm256_setzero_ps();
    let mut acc6 = _mm256_setzero_ps();
    let mut acc7 = _mm256_setzero_ps();
    let mut acc8 = _mm256_setzero_ps();

    let mut j = 0;
    while j < matrix_len {
        sum_x64_block(
            matrix_ptr.add(j + i),
            &mut acc1,
            &mut acc2,
            &mut acc3,
            &mut acc4,
            &mut acc5,
            &mut acc6,
            &mut acc7,
            &mut acc8,
        );

        j += dims;
    }

    _mm256_storeu_ps(results_ptr.add(i), acc1);
    _mm256_storeu_ps(results_ptr.add(i + 8), acc2);
    _mm256_storeu_ps(results_ptr.add(i + 16), acc3);
    _mm256_storeu_ps(results_ptr.add(i + 24), acc4);
    _mm256_storeu_ps(results_ptr.add(i + 32), acc5);
    _mm256_storeu_ps(results_ptr.add(i + 40), acc6);
    _mm256_storeu_ps(results_ptr.add(i + 48), acc7);
    _mm256_storeu_ps(results_ptr.add(i + 56), acc8);
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn sum_x64_block(
    x: *const f32,
    acc1: &mut __m256,
    acc2: &mut __m256,
    acc3: &mut __m256,
    acc4: &mut __m256,
    acc5: &mut __m256,
    acc6: &mut __m256,
    acc7: &mut __m256,
    acc8: &mut __m256,
) {
    let [x1, x2, x3, x4] = offsets_avx2_ps::<CHUNK_0>(x);
    let [x5, x6, x7, x8] = offsets_avx2_ps::<CHUNK_1>(x);

    let x1 = _mm256_loadu_ps(x1);
    let x2 = _mm256_loadu_ps(x2);
    let x3 = _mm256_loadu_ps(x3);
    let x4 = _mm256_loadu_ps(x4);
    let x5 = _mm256_loadu_ps(x5);
    let x6 = _mm256_loadu_ps(x6);
    let x7 = _mm256_loadu_ps(x7);
    let x8 = _mm256_loadu_ps(x8);

    *acc1 = _mm256_add_ps(*acc1, x1);
    *acc2 = _mm256_add_ps(*acc2, x2);
    *acc3 = _mm256_add_ps(*acc3, x3);
    *acc4 = _mm256_add_ps(*acc4, x4);
    *acc5 = _mm256_add_ps(*acc5, x5);
    *acc6 = _mm256_add_ps(*acc6, x6);
    *acc7 = _mm256_add_ps(*acc7, x7);
    *acc8 = _mm256_add_ps(*acc8, x8);
}

#[cfg(test)]
mod tests {
    use ndarray::Axis;

    use super::*;
    use crate::test_utils::{assert_is_close, get_sample_vectors};

    #[test]
    fn test_xconst_nofma_sum() {
        let (x, _) = get_sample_vectors(768);
        let sum = unsafe { f32_xconst_avx2_nofma_sum_horizontal::<768>(&x) };
        assert_is_close(sum, x.iter().sum::<f32>());
    }

    #[test]
    fn test_xany_nofma_sum() {
        let (x, _) = get_sample_vectors(131);
        let sum = unsafe { f32_xany_avx2_nofma_sum_horizontal(&x) };
        assert_is_close(sum, x.iter().sum::<f32>());
    }

    #[test]
    fn test_xconst_nofma_sum_vertical() {
        let (matrix, _) = get_sample_vectors::<f32>(512 * 25);

        let arr = ndarray::Array2::from_shape_vec((25, 512), matrix.clone()).unwrap();
        let result = arr.sum_axis(Axis(0)).to_vec();

        let mut output = vec![0.0; 512];
        unsafe { f32_xconst_avx2_nofma_sum_vertical::<512>(&matrix, &mut output) };
        assert_eq!(output, result);
    }

    #[test]
    fn test_xany_nofma_sum_vertical() {
        let (matrix, _) = get_sample_vectors::<f32>(537 * 25);

        let arr = ndarray::Array2::from_shape_vec((25, 537), matrix.clone()).unwrap();
        let result = arr.sum_axis(Axis(0)).to_vec();

        let mut output = vec![0.0; 537];
        unsafe { f32_xany_avx2_nofma_sum_vertical(&matrix, &mut output) };
        assert_eq!(output, result);
    }
}
