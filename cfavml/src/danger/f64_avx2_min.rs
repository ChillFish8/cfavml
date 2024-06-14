use core::arch::x86_64::*;
use core::mem;

use crate::danger::{offsets_avx2_pd, CHUNK_0, CHUNK_1};

#[target_feature(enable = "avx2")]
#[inline]
/// Computes the horizontal minimum of the given vector that is `[f64; DIMS]`.
///
/// # Safety
///
/// `DIMS` **MUST** be a multiple of `32`, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// This method assumes AVX2 instructions are available, if this method is executed
/// on non-AVX2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xconst_avx2_nofma_min_horizontal<const DIMS: usize>(
    arr: &[f64],
) -> f64 {
    debug_assert_eq!(arr.len(), DIMS, "Array length must match DIMS");
    debug_assert_eq!(DIMS % 32, 0, "DIMS must be a multiple of 32");

    let arr = arr.as_ptr();

    let mut acc1 = _mm256_set1_pd(f64::INFINITY);
    let mut acc2 = _mm256_set1_pd(f64::INFINITY);
    let mut acc3 = _mm256_set1_pd(f64::INFINITY);
    let mut acc4 = _mm256_set1_pd(f64::INFINITY);
    let mut acc5 = _mm256_set1_pd(f64::INFINITY);
    let mut acc6 = _mm256_set1_pd(f64::INFINITY);
    let mut acc7 = _mm256_set1_pd(f64::INFINITY);
    let mut acc8 = _mm256_set1_pd(f64::INFINITY);

    let mut i = 0;
    while i < DIMS {
        let [x1, x2, x3, x4] = offsets_avx2_pd::<CHUNK_0>(arr.add(i));
        let [x5, x6, x7, x8] = offsets_avx2_pd::<CHUNK_1>(arr.add(i));

        let x1 = _mm256_loadu_pd(x1);
        let x2 = _mm256_loadu_pd(x2);
        let x3 = _mm256_loadu_pd(x3);
        let x4 = _mm256_loadu_pd(x4);
        let x5 = _mm256_loadu_pd(x5);
        let x6 = _mm256_loadu_pd(x6);
        let x7 = _mm256_loadu_pd(x7);
        let x8 = _mm256_loadu_pd(x8);

        acc1 = _mm256_min_pd(acc1, x1);
        acc2 = _mm256_min_pd(acc2, x2);
        acc3 = _mm256_min_pd(acc3, x3);
        acc4 = _mm256_min_pd(acc4, x4);
        acc5 = _mm256_min_pd(acc5, x5);
        acc6 = _mm256_min_pd(acc6, x6);
        acc7 = _mm256_min_pd(acc7, x7);
        acc8 = _mm256_min_pd(acc8, x8);

        i += 32;
    }

    acc1 = _mm256_min_pd(acc1, acc2);
    acc3 = _mm256_min_pd(acc3, acc4);
    acc5 = _mm256_min_pd(acc5, acc6);
    acc7 = _mm256_min_pd(acc7, acc8);

    acc1 = _mm256_min_pd(acc1, acc3);
    acc5 = _mm256_min_pd(acc5, acc7);

    acc1 = _mm256_min_pd(acc1, acc5);

    let unpacked = mem::transmute::<__m256d, [f64; 4]>(acc1);

    // This is technically not the full SIMD way of doing this, but it is simpler,
    // and I am not convinced this really has a significant performance impact to warrant
    // the extra work needed to maintain it in the future.
    let mut min = f64::INFINITY;
    for x in unpacked {
        min = min.min(x);
    }

    min
}

#[target_feature(enable = "avx2")]
#[allow(unused)]
#[inline]
/// Computes the vertical minimum of the given vector that is `[[f64; DIMS]; N]`.
///
/// # Safety
///
/// `DIMS` **MUST** be a multiple of `32`, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// This method assumes AVX2 instructions are available, if this method is executed
/// on non-AVX2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xconst_avx2_nofma_min_vertical<const DIMS: usize>(
    matrix: &[f64],
    output: &mut [f64],
) {
    debug_assert_eq!(DIMS % 32, 0, "DIMS must be a multiple of 32");

    let matrix_len = matrix.len();
    let matrix_ptr = matrix.as_ptr();
    let min_values_ptr = output.as_mut_ptr();

    // We work our way horizontally by taking steps of 64 and finding
    // the min of for each of the lanes vertically through the matrix.
    let mut i = 0;
    while i < DIMS {
        min_vertical_component(
            i,
            matrix_ptr,
            matrix_len,
            min_values_ptr,
            DIMS,
        );
        i += 32;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
/// Computes the horizontal minimum of the given vector that is `[f64; N]`.
///
/// # Safety
///
/// This method assumes AVX2 instructions are available, if this method is executed
/// on non-AVX2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xany_avx2_nofma_min_horizontal(arr: &[f64]) -> f64 {
    let len = arr.len();
    let offset_from = len % 32;

    let mut min = f64::INFINITY;

    let mut acc1 = _mm256_set1_pd(f64::INFINITY);
    let mut acc2 = _mm256_set1_pd(f64::INFINITY);
    let mut acc3 = _mm256_set1_pd(f64::INFINITY);
    let mut acc4 = _mm256_set1_pd(f64::INFINITY);
    let mut acc5 = _mm256_set1_pd(f64::INFINITY);
    let mut acc6 = _mm256_set1_pd(f64::INFINITY);
    let mut acc7 = _mm256_set1_pd(f64::INFINITY);
    let mut acc8 = _mm256_set1_pd(f64::INFINITY);

    let arr_ptr = arr.as_ptr();

    let mut i = 0;
    while i < (len - offset_from) {
        let [x1, x2, x3, x4] = offsets_avx2_pd::<CHUNK_0>(arr_ptr.add(i));
        let [x5, x6, x7, x8] = offsets_avx2_pd::<CHUNK_1>(arr_ptr.add(i));

        let x1 = _mm256_loadu_pd(x1);
        let x2 = _mm256_loadu_pd(x2);
        let x3 = _mm256_loadu_pd(x3);
        let x4 = _mm256_loadu_pd(x4);
        let x5 = _mm256_loadu_pd(x5);
        let x6 = _mm256_loadu_pd(x6);
        let x7 = _mm256_loadu_pd(x7);
        let x8 = _mm256_loadu_pd(x8);

        acc1 = _mm256_min_pd(acc1, x1);
        acc2 = _mm256_min_pd(acc2, x2);
        acc3 = _mm256_min_pd(acc3, x3);
        acc4 = _mm256_min_pd(acc4, x4);
        acc5 = _mm256_min_pd(acc5, x5);
        acc6 = _mm256_min_pd(acc6, x6);
        acc7 = _mm256_min_pd(acc7, x7);
        acc8 = _mm256_min_pd(acc8, x8);

        i += 32;
    }

    if offset_from != 0 {
        let tail = offset_from % 4;

        while i < (len - tail) {
            let x = _mm256_loadu_pd(arr_ptr.add(i));
            acc1 = _mm256_min_pd(acc1, x);

            i += 4;
        }

        for n in i..len {
            let x = *arr.get_unchecked(n);
            min = min.min(x);
        }
    }

    acc1 = _mm256_min_pd(acc1, acc2);
    acc3 = _mm256_min_pd(acc3, acc4);
    acc5 = _mm256_min_pd(acc5, acc6);
    acc7 = _mm256_min_pd(acc7, acc8);

    acc1 = _mm256_min_pd(acc1, acc3);
    acc5 = _mm256_min_pd(acc5, acc7);

    acc1 = _mm256_min_pd(acc1, acc5);

    let unpacked = mem::transmute::<__m256d, [f64; 4]>(acc1);

    // This is technically not the full SIMD way of doing this, but it is simpler,
    // and I am not convinced this really has a significant performance impact to warrant
    // the extra work needed to maintain it in the future.
    for x in unpacked {
        min = min.min(x);
    }

    min
}

#[target_feature(enable = "avx2")]
#[allow(unused)]
#[inline]
/// Computes the vertical minimum of the given vector that is `[[f64; N]; N2]`.
///
/// # Safety
///
/// The size of each array in the matrix must be equal otherwise out of bounds
/// access can occur.
///
/// This method assumes AVX2 instructions are available, if this method is executed
/// on non-AVX2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xany_avx2_nofma_min_vertical(
    matrix: &[f64],
    output: &mut [f64],
) {
    let dims = output.len();

    debug_assert_eq!(matrix.len() % dims, 0, "Matrix must be a multiple of dims");

    let matrix_len = matrix.len();
    let matrix_ptr = matrix.as_ptr();

    let offset_from = dims % 32;

    let min_values_ptr = output.as_mut_ptr();

    // We work our way horizontally by taking steps of 64 and finding
    // the min of for each of the lanes vertically through the matrix.
    // TODO: I am unsure how hard this is on the cache or if there is a smarter
    //       way to write this.
    let mut i = 0;
    while i < (dims - offset_from) {
        min_vertical_component(
            i,
            matrix_ptr,
            matrix_len,
            min_values_ptr,
            dims,
        );

        i += 32;
    }

    if offset_from != 0 {
        let tail = offset_from % 4;

        while i < (dims - tail) {
            let mut acc = _mm256_set1_pd(f64::INFINITY);

            let mut j = 0;
            while j < matrix_len {
                let x = _mm256_loadu_pd(matrix_ptr.add(j + i));
                acc = _mm256_min_pd(acc, x);

                j += dims;
            }

            _mm256_storeu_pd(min_values_ptr.add(i), acc);

            i += 4;
        }

        while i < dims {
            let mut min = f64::INFINITY;

            let mut j = 0;
            while j < matrix_len {
                let x = *matrix.get_unchecked(j + i);
                min = min.min(x);

                j += dims;
            }

            *output.get_unchecked_mut(i) = min;

            i += 1;
        }
    }
}


#[inline(always)]
unsafe fn min_vertical_component(
    i: usize,
    matrix_ptr: *const f64,
    matrix_len: usize,
    results_ptr: *mut f64,
    dims: usize,
) {
    let mut acc1 = _mm256_set1_pd(f64::INFINITY);
    let mut acc2 = _mm256_set1_pd(f64::INFINITY);
    let mut acc3 = _mm256_set1_pd(f64::INFINITY);
    let mut acc4 = _mm256_set1_pd(f64::INFINITY);
    let mut acc5 = _mm256_set1_pd(f64::INFINITY);
    let mut acc6 = _mm256_set1_pd(f64::INFINITY);
    let mut acc7 = _mm256_set1_pd(f64::INFINITY);
    let mut acc8 = _mm256_set1_pd(f64::INFINITY);

    // Vertical max of the 32 elements.
    let mut j = 0;
    while j < matrix_len {
        let [x1, x2, x3, x4] = offsets_avx2_pd::<CHUNK_0>(matrix_ptr.add(j + i));
        let [x5, x6, x7, x8] = offsets_avx2_pd::<CHUNK_1>(matrix_ptr.add(j + i));

        let x1 = _mm256_loadu_pd(x1);
        let x2 = _mm256_loadu_pd(x2);
        let x3 = _mm256_loadu_pd(x3);
        let x4 = _mm256_loadu_pd(x4);
        let x5 = _mm256_loadu_pd(x5);
        let x6 = _mm256_loadu_pd(x6);
        let x7 = _mm256_loadu_pd(x7);
        let x8 = _mm256_loadu_pd(x8);

        acc1 = _mm256_min_pd(acc1, x1);
        acc2 = _mm256_min_pd(acc2, x2);
        acc3 = _mm256_min_pd(acc3, x3);
        acc4 = _mm256_min_pd(acc4, x4);
        acc5 = _mm256_min_pd(acc5, x5);
        acc6 = _mm256_min_pd(acc6, x6);
        acc7 = _mm256_min_pd(acc7, x7);
        acc8 = _mm256_min_pd(acc8, x8);

        j += dims;
    }

    _mm256_storeu_pd(results_ptr.add(i), acc1);
    _mm256_storeu_pd(results_ptr.add(i + 4), acc2);
    _mm256_storeu_pd(results_ptr.add(i + 8), acc3);
    _mm256_storeu_pd(results_ptr.add(i + 12), acc4);
    _mm256_storeu_pd(results_ptr.add(i + 16), acc5);
    _mm256_storeu_pd(results_ptr.add(i + 20), acc6);
    _mm256_storeu_pd(results_ptr.add(i + 24), acc7);
    _mm256_storeu_pd(results_ptr.add(i + 28), acc8);
}

#[cfg(test)]
mod tests {
    use ndarray::Axis;
    use super::*;
    use crate::test_utils::get_sample_vectors;

    #[test]
    fn test_xconst_nofma_min_horizontal() {
        let (x, _) = get_sample_vectors(512);
        let min = unsafe { f64_xconst_avx2_nofma_min_horizontal::<512>(&x) };
        assert_eq!(min, x.iter().fold(f64::INFINITY, |acc, v| acc.min(*v)));
    }

    #[test]
    fn test_xconst_nofma_min_vertical() {
        let (matrix, _) = get_sample_vectors::<f64>(512 * 25);

        let arr = ndarray::Array2::from_shape_vec((25, 512), matrix.clone()).unwrap();
        let expected_vertical_min = arr
            .map_axis(Axis(0), |a| {
                a.iter().min_by(|a, b| a.total_cmp(b)).copied().unwrap()
            })
            .to_vec();

        let mut max = vec![f64::INFINITY; 512];
        unsafe { f64_xconst_avx2_nofma_min_vertical::<512>(&matrix, &mut max) };
        assert_eq!(max, expected_vertical_min);
    }

    #[test]
    fn test_xany_nofma_min_horizontal() {
        let (x, _) = get_sample_vectors(793);
        let min = unsafe { f64_xany_avx2_nofma_min_horizontal(&x) };
        assert_eq!(min, x.iter().fold(f64::INFINITY, |acc, v| acc.min(*v)));
    }

    #[test]
    fn test_xany_nofma_min_vertical() {
        let (matrix, _) = get_sample_vectors::<f64>(537 * 25);

        let arr = ndarray::Array2::from_shape_vec((25, 537), matrix.clone()).unwrap();
        let expected_vertical_min = arr
            .map_axis(Axis(0), |a| {
                a.iter().min_by(|a, b| a.total_cmp(b)).copied().unwrap()
            })
            .to_vec();

        let mut max = vec![f64::INFINITY; 537];
        unsafe { f64_xany_avx2_nofma_min_vertical(&matrix, &mut max) };
        assert_eq!(max, expected_vertical_min);
    }
}
