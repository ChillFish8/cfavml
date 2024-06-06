use std::arch::x86_64::*;
use std::{mem, ptr};

use crate::danger::{
    copy_masked_avx512_ps_register_to,
    load_one_variable_size_avx512_ps,
    offsets_avx512_ps,
    sum_avx512_x8_ps,
    CHUNK_0,
    CHUNK_1,
};

#[target_feature(enable = "avx512f")]
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
/// Vectors **MUST** be a multiple of `64`, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// This method assumes AVX512 instructions are available, if this method is executed
/// on non-AVX512 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xconst_avx512_nofma_sum_horizontal<const DIMS: usize>(
    x: &[f32],
) -> f32 {
    debug_assert_eq!(DIMS % 128, 0, "DIMS must be a multiple of 128");
    debug_assert_eq!(x.len(), DIMS);

    let x = x.as_ptr();

    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut acc4 = _mm512_setzero_ps();
    let mut acc5 = _mm512_setzero_ps();
    let mut acc6 = _mm512_setzero_ps();
    let mut acc7 = _mm512_setzero_ps();
    let mut acc8 = _mm512_setzero_ps();

    let mut i = 0;
    while i < DIMS {
        sum_x128_block(
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

        i += 128;
    }

    sum_avx512_x8_ps(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8)
}

#[target_feature(enable = "avx512f")]
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
/// This method assumes AVX512 instructions are available, if this method is executed
/// on non-AVX512 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xany_avx512_nofma_sum_horizontal(x: &[f32]) -> f32 {
    let len = x.len();
    let offset_from = len % 128;

    let x = x.as_ptr();

    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut acc4 = _mm512_setzero_ps();
    let mut acc5 = _mm512_setzero_ps();
    let mut acc6 = _mm512_setzero_ps();
    let mut acc7 = _mm512_setzero_ps();
    let mut acc8 = _mm512_setzero_ps();

    let mut i = 0;
    while i < (len - offset_from) {
        sum_x128_block(
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

        i += 128;
    }

    while i < len {
        let n = len - i;

        let x = load_one_variable_size_avx512_ps(x.add(i), n);
        acc1 = _mm512_add_ps(acc1, x);

        i += 16;
    }

    sum_avx512_x8_ps(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8)
}

#[allow(unused)]
#[target_feature(enable = "avx512f")]
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
/// `DIMS` **MUST** be a multiple of `128`, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// All vectors within the matrix must also `DIMS` in length.
///
/// This method assumes AVX512 instructions are available, if this method is executed
/// on non-AVX512 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xconst_avx512_nofma_sum_vertical<const DIMS: usize>(
    matrix: &[&[f32]],
) -> Vec<f32> {
    debug_assert_eq!(DIMS % 128, 0, "DIMS must be a multiple of 128");

    let mut results = vec![0.0; DIMS];
    let results_ptr = results.as_mut_ptr();

    let mut i = 0;
    while i < DIMS {
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();
        let mut acc4 = _mm512_setzero_ps();
        let mut acc5 = _mm512_setzero_ps();
        let mut acc6 = _mm512_setzero_ps();
        let mut acc7 = _mm512_setzero_ps();
        let mut acc8 = _mm512_setzero_ps();

        for m in 0..matrix.len() {
            let arr = *matrix.get_unchecked(m);
            debug_assert_eq!(arr.len(), DIMS);
            let arr = arr.as_ptr();

            sum_x128_block(
                arr.add(i),
                &mut acc1,
                &mut acc2,
                &mut acc3,
                &mut acc4,
                &mut acc5,
                &mut acc6,
                &mut acc7,
                &mut acc8,
            );
        }

        let merged = [acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8];

        let result = mem::transmute::<[__m512; 8], [f32; 128]>(merged);
        ptr::copy_nonoverlapping(result.as_ptr(), results_ptr.add(i), result.len());

        i += 128;
    }

    results
}

#[allow(unused)]
#[target_feature(enable = "avx512f")]
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
/// This method assumes AVX512 instructions are available, if this method is executed
/// on non-AVX512 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xany_avx512_nofma_sum_vertical(matrix: &[&[f32]]) -> Vec<f32> {
    let len = matrix[0].len();
    let offset_from = len % 128;

    let mut results = vec![0.0; len];
    let results_ptr = results.as_mut_ptr();

    let mut i = 0;
    while i < (len - offset_from) {
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();
        let mut acc4 = _mm512_setzero_ps();
        let mut acc5 = _mm512_setzero_ps();
        let mut acc6 = _mm512_setzero_ps();
        let mut acc7 = _mm512_setzero_ps();
        let mut acc8 = _mm512_setzero_ps();

        for m in 0..matrix.len() {
            let arr = *matrix.get_unchecked(m);
            debug_assert_eq!(arr.len(), len);
            let arr = arr.as_ptr();

            sum_x128_block(
                arr.add(i),
                &mut acc1,
                &mut acc2,
                &mut acc3,
                &mut acc4,
                &mut acc5,
                &mut acc6,
                &mut acc7,
                &mut acc8,
            );
        }

        let merged = [acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8];

        let result = mem::transmute::<[__m512; 8], [f32; 128]>(merged);
        ptr::copy_nonoverlapping(result.as_ptr(), results_ptr.add(i), result.len());

        i += 128;
    }

    while i < len {
        let n = len - i;

        let mut acc = _mm512_setzero_ps();

        for m in 0..matrix.len() {
            let arr = *matrix.get_unchecked(m);
            debug_assert_eq!(arr.len(), len);

            let arr = arr.as_ptr();
            let x = load_one_variable_size_avx512_ps(arr.add(i), n);
            acc = _mm512_add_ps(acc, x);
        }

        copy_masked_avx512_ps_register_to(results_ptr.add(i), acc, n);

        i += 16;
    }

    results
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn sum_x128_block(
    x: *const f32,
    acc1: &mut __m512,
    acc2: &mut __m512,
    acc3: &mut __m512,
    acc4: &mut __m512,
    acc5: &mut __m512,
    acc6: &mut __m512,
    acc7: &mut __m512,
    acc8: &mut __m512,
) {
    let [x1, x2, x3, x4] = offsets_avx512_ps::<CHUNK_0>(x);
    let [x5, x6, x7, x8] = offsets_avx512_ps::<CHUNK_1>(x);

    let x1 = _mm512_loadu_ps(x1);
    let x2 = _mm512_loadu_ps(x2);
    let x3 = _mm512_loadu_ps(x3);
    let x4 = _mm512_loadu_ps(x4);
    let x5 = _mm512_loadu_ps(x5);
    let x6 = _mm512_loadu_ps(x6);
    let x7 = _mm512_loadu_ps(x7);
    let x8 = _mm512_loadu_ps(x8);

    *acc1 = _mm512_add_ps(*acc1, x1);
    *acc2 = _mm512_add_ps(*acc2, x2);
    *acc3 = _mm512_add_ps(*acc3, x3);
    *acc4 = _mm512_add_ps(*acc4, x4);
    *acc5 = _mm512_add_ps(*acc5, x5);
    *acc6 = _mm512_add_ps(*acc6, x6);
    *acc7 = _mm512_add_ps(*acc7, x7);
    *acc8 = _mm512_add_ps(*acc8, x8);
}

#[cfg(all(test, target_feature = "avx512f"))]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{assert_is_close, get_sample_vectors};

    #[test]
    fn test_xconst_nofma_sum() {
        let (x, _) = get_sample_vectors(768);
        let sum = unsafe { f32_xconst_avx512_nofma_sum_horizontal::<768>(&x) };
        assert_is_close(sum, x.iter().sum::<f32>());
    }

    #[test]
    fn test_xany_nofma_sum() {
        let (x, _) = get_sample_vectors(131);
        let sum = unsafe { f32_xany_avx512_nofma_sum_horizontal(&x) };
        assert_is_close(sum, x.iter().sum::<f32>());
    }

    #[test]
    fn test_xconst_nofma_sum_vertical() {
        let mut matrix = Vec::new();
        for _ in 0..25 {
            let (x, _) = get_sample_vectors(512);
            matrix.push(x);
        }

        let matrix_view = matrix.iter().map(|v| v.as_ref()).collect::<Vec<&[f32]>>();

        let mut expected_vertical_sum = vec![0.0; 512];
        for i in 0..512 {
            let mut sum = 0.0;
            for arr in matrix.iter() {
                sum += arr[i];
            }
            expected_vertical_sum[i] = sum;
        }

        let sum = unsafe { f32_xconst_avx512_nofma_sum_vertical::<512>(&matrix_view) };
        assert_eq!(sum, expected_vertical_sum);
    }

    #[test]
    fn test_xany_nofma_sum_vertical() {
        let mut matrix = Vec::new();
        for _ in 0..25 {
            let (x, _) = get_sample_vectors(537);
            matrix.push(x);
        }

        let matrix_view = matrix.iter().map(|v| v.as_ref()).collect::<Vec<&[f32]>>();

        let mut expected_vertical_sum = vec![0.0; 537];
        for i in 0..537 {
            let mut sum = 0.0;
            for arr in matrix.iter() {
                sum += arr[i];
            }
            expected_vertical_sum[i] = sum;
        }

        let sum = unsafe { f32_xany_avx512_nofma_sum_vertical(&matrix_view) };
        assert_eq!(sum, expected_vertical_sum);
    }
}
