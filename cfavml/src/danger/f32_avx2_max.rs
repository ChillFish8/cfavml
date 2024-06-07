use core::arch::x86_64::*;
use core::mem;

use crate::danger::{offsets_avx2_ps, CHUNK_0, CHUNK_1};

#[target_feature(enable = "avx2")]
#[inline]
/// Computes the horizontal maximum of the given vector that is `[f32; DIMS]`.
///
/// # Safety
///
/// `DIMS` **MUST** be a multiple of `64`, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// This method assumes AVX2 instructions are available, if this method is executed
/// on non-AVX2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xconst_avx2_nofma_max_horizontal<const DIMS: usize>(
    arr: &[f32],
) -> f32 {
    debug_assert_eq!(arr.len(), DIMS, "Array length must match DIMS");
    debug_assert_eq!(DIMS % 64, 0, "DIMS must be a multiple of 64");

    let arr = arr.as_ptr();

    let mut acc1 = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut acc2 = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut acc3 = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut acc4 = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut acc5 = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut acc6 = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut acc7 = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut acc8 = _mm256_set1_ps(f32::NEG_INFINITY);

    let mut i = 0;
    while i < DIMS {
        let [x1, x2, x3, x4] = offsets_avx2_ps::<CHUNK_0>(arr.add(i));
        let [x5, x6, x7, x8] = offsets_avx2_ps::<CHUNK_1>(arr.add(i));

        let x1 = _mm256_loadu_ps(x1);
        let x2 = _mm256_loadu_ps(x2);
        let x3 = _mm256_loadu_ps(x3);
        let x4 = _mm256_loadu_ps(x4);
        let x5 = _mm256_loadu_ps(x5);
        let x6 = _mm256_loadu_ps(x6);
        let x7 = _mm256_loadu_ps(x7);
        let x8 = _mm256_loadu_ps(x8);

        acc1 = _mm256_max_ps(acc1, x1);
        acc2 = _mm256_max_ps(acc2, x2);
        acc3 = _mm256_max_ps(acc3, x3);
        acc4 = _mm256_max_ps(acc4, x4);
        acc5 = _mm256_max_ps(acc5, x5);
        acc6 = _mm256_max_ps(acc6, x6);
        acc7 = _mm256_max_ps(acc7, x7);
        acc8 = _mm256_max_ps(acc8, x8);

        i += 64;
    }

    acc1 = _mm256_max_ps(acc1, acc2);
    acc3 = _mm256_max_ps(acc3, acc4);
    acc5 = _mm256_max_ps(acc5, acc6);
    acc7 = _mm256_max_ps(acc7, acc8);

    acc1 = _mm256_max_ps(acc1, acc3);
    acc5 = _mm256_max_ps(acc5, acc7);

    acc1 = _mm256_max_ps(acc1, acc5);

    let unpacked = mem::transmute::<__m256, [f32; 8]>(acc1);

    // This is technically not the full SIMD way of doing this, but it is simpler,
    // and I am not convinced this really has a significant performance impact to warrant
    // the extra work needed to maintain it in the future.
    let mut max = f32::NEG_INFINITY;
    for x in unpacked {
        max = max.max(x);
    }

    max
}

#[target_feature(enable = "avx2")]
#[allow(unused)]
#[inline]
/// Computes the vertical maximum of the given vector that is `[[f32; DIMS]; N]`.
///
/// Matrix is computed assuming a row-major element order and as a 2D matrix.
///
/// # Safety
///
/// `DIMS` **MUST** be a multiple of `64`, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// `output` & rows of the matrix must also be `DIMS` in length.
///
/// This method assumes AVX2 instructions are available, if this method is executed
/// on non-AVX2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xconst_avx2_nofma_max_vertical<const DIMS: usize>(
    matrix: &[f32],
    output: &mut [f32],
) {
    debug_assert_eq!(DIMS % 64, 0, "DIMS must be a multiple of 64");
    debug_assert_eq!(matrix.len() % DIMS, 0, "Matrix shape must be `[[f32; DIMS]; N]`");
    debug_assert_eq!(output.len(), DIMS, "Output buffer must be DIMS in size");

    let matrix_len = matrix.len();
    let matrix_ptr = matrix.as_ptr();

    let max_values_ptr = output.as_mut_ptr();

    // We work our way horizontally by taking steps of 64 and finding
    // the max of for each of the lanes vertically through the matrix.
    let mut i = 0;
    while i < DIMS {
        vertical_max_component(
            i,
            matrix_ptr,
            matrix_len,
            max_values_ptr,
            DIMS,
        );

        i += 64;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
/// Computes the horizontal maximum of the given vector that is `[f32; N]`.
///
/// # Safety
///
/// This method assumes AVX2 instructions are available, if this method is executed
/// on non-AVX2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xany_avx2_nofma_max_horizontal(arr: &[f32]) -> f32 {
    let len = arr.len();
    let offset_from = len % 64;

    let mut max = f32::NEG_INFINITY;

    let mut acc1 = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut acc2 = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut acc3 = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut acc4 = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut acc5 = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut acc6 = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut acc7 = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut acc8 = _mm256_set1_ps(f32::NEG_INFINITY);

    let arr_ptr = arr.as_ptr();

    let mut i = 0;
    while i < (len - offset_from) {
        let [x1, x2, x3, x4] = offsets_avx2_ps::<CHUNK_0>(arr_ptr.add(i));
        let [x5, x6, x7, x8] = offsets_avx2_ps::<CHUNK_1>(arr_ptr.add(i));

        let x1 = _mm256_loadu_ps(x1);
        let x2 = _mm256_loadu_ps(x2);
        let x3 = _mm256_loadu_ps(x3);
        let x4 = _mm256_loadu_ps(x4);
        let x5 = _mm256_loadu_ps(x5);
        let x6 = _mm256_loadu_ps(x6);
        let x7 = _mm256_loadu_ps(x7);
        let x8 = _mm256_loadu_ps(x8);

        acc1 = _mm256_max_ps(acc1, x1);
        acc2 = _mm256_max_ps(acc2, x2);
        acc3 = _mm256_max_ps(acc3, x3);
        acc4 = _mm256_max_ps(acc4, x4);
        acc5 = _mm256_max_ps(acc5, x5);
        acc6 = _mm256_max_ps(acc6, x6);
        acc7 = _mm256_max_ps(acc7, x7);
        acc8 = _mm256_max_ps(acc8, x8);

        i += 64;
    }

    if offset_from != 0 {
        let tail = offset_from % 8;

        while i < (len - tail) {
            let x = _mm256_loadu_ps(arr_ptr.add(i));
            acc1 = _mm256_max_ps(acc1, x);

            i += 8;
        }

        for n in i..len {
            let x = *arr.get_unchecked(n);
            max = max.max(x);
        }
    }

    acc1 = _mm256_max_ps(acc1, acc2);
    acc3 = _mm256_max_ps(acc3, acc4);
    acc5 = _mm256_max_ps(acc5, acc6);
    acc7 = _mm256_max_ps(acc7, acc8);

    acc1 = _mm256_max_ps(acc1, acc3);
    acc5 = _mm256_max_ps(acc5, acc7);

    acc1 = _mm256_max_ps(acc1, acc5);

    let unpacked = mem::transmute::<__m256, [f32; 8]>(acc1);

    // This is technically not the full SIMD way of doing this, but it is simpler,
    // and I am not convinced this really has a significant performance impact to warrant
    // the extra work needed to maintain it in the future.
    for x in unpacked {
        max = max.max(x);
    }

    max
}

#[target_feature(enable = "avx2")]
#[allow(unused)]
#[inline]
/// Computes the vertical maximum of the given vector that is `[[f32; N]; N2]`.
///
/// Matrix is computed assuming a row-major element order and as a 2D matrix.
///
/// The dimensions of the matrix
///
/// # Safety
///
/// The size of each array in the matrix must be equal otherwise out of bounds
/// access can occur.
///
/// This method assumes AVX2 instructions are available, if this method is executed
/// on non-AVX2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xany_avx2_nofma_max_vertical(matrix: &[f32], output: &mut [f32]) {
    let dims = output.len();
    let offset_from = dims % 64;

    let matrix_len = matrix.len();
    let matrix_ptr = matrix.as_ptr();

    let max_values_ptr = output.as_mut_ptr();
    let max_values = output;

    // We work our way horizontally by taking steps of 64 and finding
    // the max of for each of the lanes vertically through the matrix.
    let mut i = 0;
    while i < (dims - offset_from) {
        vertical_max_component(
            i,
            matrix_ptr,
            matrix_len,
            max_values_ptr,
            dims,
        );

        i += 64;
    }

    if offset_from != 0 {
        let tail = offset_from % 8;

        while i < (dims - tail) {
            let mut acc = _mm256_set1_ps(f32::NEG_INFINITY);

            let mut j = 0;
            while j < matrix_len {
                let x = _mm256_loadu_ps(matrix_ptr.add(j + i));
                acc = _mm256_max_ps(acc, x);

                j += dims;
            }

            _mm256_storeu_ps(max_values_ptr.add(i), acc);

            i += 8;
        }

        for i in i..dims {
            let mut max = f32::NEG_INFINITY;
            let mut j = 0;
            while j < matrix_len {
                let x = *matrix.get_unchecked(j + i);
                max = max.max(x);

                j += dims;
            }

            *max_values.get_unchecked_mut(i) = max;
        }
    }
}

#[inline(always)]
unsafe fn vertical_max_component(
    i: usize,
    matrix_ptr: *const f32,
    matrix_len: usize,
    results_ptr: *mut f32,
    dims: usize,
) {
    // TODO: I am unsure how hard this is on the cache or if there is a smarter
    //       way to write this.

    let mut acc1 = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut acc2 = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut acc3 = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut acc4 = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut acc5 = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut acc6 = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut acc7 = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut acc8 = _mm256_set1_ps(f32::NEG_INFINITY);

    // Vertical max of the 64 elements.
    let mut j = 0;
    while j < matrix_len {
        let [x1, x2, x3, x4] = offsets_avx2_ps::<CHUNK_0>(matrix_ptr.add(j + i));
        let [x5, x6, x7, x8] = offsets_avx2_ps::<CHUNK_1>(matrix_ptr.add(j + i));

        let x1 = _mm256_loadu_ps(x1);
        let x2 = _mm256_loadu_ps(x2);
        let x3 = _mm256_loadu_ps(x3);
        let x4 = _mm256_loadu_ps(x4);
        let x5 = _mm256_loadu_ps(x5);
        let x6 = _mm256_loadu_ps(x6);
        let x7 = _mm256_loadu_ps(x7);
        let x8 = _mm256_loadu_ps(x8);

        acc1 = _mm256_max_ps(acc1, x1);
        acc2 = _mm256_max_ps(acc2, x2);
        acc3 = _mm256_max_ps(acc3, x3);
        acc4 = _mm256_max_ps(acc4, x4);
        acc5 = _mm256_max_ps(acc5, x5);
        acc6 = _mm256_max_ps(acc6, x6);
        acc7 = _mm256_max_ps(acc7, x7);
        acc8 = _mm256_max_ps(acc8, x8);

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

#[cfg(test)]
mod tests {
    use ndarray::Axis;
    use super::*;
    use crate::test_utils::get_sample_vectors;

    #[test]
    fn test_xconst_nofma_max_horizontal() {
        let (x, _) = get_sample_vectors(512);
        let max = unsafe { f32_xconst_avx2_nofma_max_horizontal::<512>(&x) };
        assert_eq!(max, x.iter().fold(f32::NEG_INFINITY, |acc, v| acc.max(*v)));
    }

    #[test]
    fn test_xconst_nofma_max_vertical() {
        let (matrix, _) = get_sample_vectors::<f32>(512 * 25);

        let arr = ndarray::Array2::from_shape_vec((25, 512), matrix.clone()).unwrap();
        let expected_vertical_max = arr.map_axis(Axis(0), |a| a.iter().max_by(|a, b| a.total_cmp(b)).copied().unwrap()).to_vec();

        let mut max = vec![f32::NEG_INFINITY; 512];
        unsafe { f32_xconst_avx2_nofma_max_vertical::<512>(&matrix, &mut max) };
        assert_eq!(max, expected_vertical_max);
    }

    #[test]
    fn test_xany_nofma_max_horizontal() {
        let (x, _) = get_sample_vectors(793);
        let max = unsafe { f32_xany_avx2_nofma_max_horizontal(&x) };
        assert_eq!(max, x.iter().fold(f32::NEG_INFINITY, |acc, v| acc.max(*v)));
    }

    #[test]
    fn test_xany_nofma_max_vertical() {
        let (matrix, _) = get_sample_vectors::<f32>(537 * 25);

        let arr = ndarray::Array2::from_shape_vec((25, 537), matrix.clone()).unwrap();
        let expected_vertical_max = arr.map_axis(Axis(0), |a| a.iter().max_by(|a, b| a.total_cmp(b)).copied().unwrap()).to_vec();

        let mut max = vec![f32::NEG_INFINITY; 537];
        unsafe { f32_xany_avx2_nofma_max_vertical(&matrix, &mut max) };
        assert_eq!(max, expected_vertical_max);
    }
}
