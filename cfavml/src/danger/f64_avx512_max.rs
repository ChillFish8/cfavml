use std::arch::x86_64::*;
use std::{mem, ptr};

use crate::danger::{
    copy_masked_avx512_pd_register_to,
    load_one_variable_size_avx512_pd,
    offsets_avx512_pd,
    CHUNK_0,
    CHUNK_1,
};

#[allow(unused)]
#[target_feature(enable = "avx512f")]
#[inline]
/// Computes the horizontal maximum of the given vector that is `[f64; DIMS]`.
///
/// # Safety
///
/// `DIMS` **MUST** be a multiple of `64`, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// This method assumes AVX512 instructions are available, if this method is executed
/// on non-AVX512 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xconst_avx512_nofma_max_horizontal<const DIMS: usize>(
    arr: &[f64],
) -> f64 {
    debug_assert_eq!(arr.len(), DIMS, "Array length must match DIMS");
    debug_assert_eq!(DIMS % 64, 0, "DIMS must be a multiple of 64");

    let arr = arr.as_ptr();

    let mut acc1 = _mm512_set1_pd(f64::NEG_INFINITY);
    let mut acc2 = _mm512_set1_pd(f64::NEG_INFINITY);
    let mut acc3 = _mm512_set1_pd(f64::NEG_INFINITY);
    let mut acc4 = _mm512_set1_pd(f64::NEG_INFINITY);
    let mut acc5 = _mm512_set1_pd(f64::NEG_INFINITY);
    let mut acc6 = _mm512_set1_pd(f64::NEG_INFINITY);
    let mut acc7 = _mm512_set1_pd(f64::NEG_INFINITY);
    let mut acc8 = _mm512_set1_pd(f64::NEG_INFINITY);

    let mut i = 0;
    while i < DIMS {
        max_by_x64_horizontal(
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

        i += 64;
    }

    rollup_max_acc(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8)
}

#[allow(unused)]
#[target_feature(enable = "avx512f")]
#[inline]
/// Computes the vertical maximum of the given vector that is `[[f64; DIMS]; N]`.
///
/// # Safety
///
/// `DIMS` **MUST** be a multiple of `64`, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// This method assumes AVX512 instructions are available, if this method is executed
/// on non-AVX512 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xconst_avx512_nofma_max_vertical<const DIMS: usize>(
    matrix: &[&[f64]],
) -> Vec<f64> {
    debug_assert_eq!(DIMS % 64, 0, "DIMS must be a multiple of 64");

    let mut max_values = vec![0.0; DIMS];
    let max_values_ptr = max_values.as_mut_ptr();

    // We work our way horizontally by taking steps of 64 and finding
    // the max of for each of the lanes vertically through the matrix.
    // TODO: I am unsure how hard this is on the cache or if there is a smarter
    //       way to write this.
    let mut i = 0;
    while i < DIMS {
        let mut acc1 = _mm512_set1_pd(f64::NEG_INFINITY);
        let mut acc2 = _mm512_set1_pd(f64::NEG_INFINITY);
        let mut acc3 = _mm512_set1_pd(f64::NEG_INFINITY);
        let mut acc4 = _mm512_set1_pd(f64::NEG_INFINITY);
        let mut acc5 = _mm512_set1_pd(f64::NEG_INFINITY);
        let mut acc6 = _mm512_set1_pd(f64::NEG_INFINITY);
        let mut acc7 = _mm512_set1_pd(f64::NEG_INFINITY);
        let mut acc8 = _mm512_set1_pd(f64::NEG_INFINITY);

        // Vertical max of the 64 elements.
        for m in 0..matrix.len() {
            let arr = *matrix.get_unchecked(m);
            let arr = arr.as_ptr();

            max_by_x64_horizontal(
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

        let result = mem::transmute::<[__m512d; 8], [f64; 64]>(merged);
        ptr::copy_nonoverlapping(result.as_ptr(), max_values_ptr.add(i), result.len());

        i += 64;
    }

    max_values
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Computes the horizontal maximum of the given vector that is `[f64; N]`.
///
/// # Safety
///
/// This method assumes AVX512 instructions are available, if this method is executed
/// on non-AVX512 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xany_avx512_nofma_max_horizontal(arr: &[f64]) -> f64 {
    let len = arr.len();
    let arr = arr.as_ptr();

    let mut acc1 = _mm512_set1_pd(f64::NEG_INFINITY);
    let mut acc2 = _mm512_set1_pd(f64::NEG_INFINITY);
    let mut acc3 = _mm512_set1_pd(f64::NEG_INFINITY);
    let mut acc4 = _mm512_set1_pd(f64::NEG_INFINITY);
    let mut acc5 = _mm512_set1_pd(f64::NEG_INFINITY);
    let mut acc6 = _mm512_set1_pd(f64::NEG_INFINITY);
    let mut acc7 = _mm512_set1_pd(f64::NEG_INFINITY);
    let mut acc8 = _mm512_set1_pd(f64::NEG_INFINITY);

    let offset_from = len % 64;

    let mut i = 0;
    while i < (len - offset_from) {
        max_by_x64_horizontal(
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

        i += 64;
    }

    while i < len {
        let n = len - i;

        if n < 8 {
            let mask = _bzhi_u32(0xFFFFFFFF, n as u32) as _;
            let x = _mm512_maskz_loadu_pd(mask, arr);
            acc1 = _mm512_mask_max_pd(acc1, mask, acc1, x);
        } else {
            let x = _mm512_loadu_pd(arr);
            acc1 = _mm512_max_pd(acc1, x);
        }

        i += 8;
    }

    rollup_max_acc(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8)
}

#[allow(unused)]
#[target_feature(enable = "avx512f")]
#[inline]
/// Computes the vertical maximum of the given vector that is `[[f64; N]; N2]`.
///
/// # Safety
///
/// The size of each array in the matrix must be equal otherwise out of bounds
/// access can occur.
///
/// This method assumes AVX512 instructions are available, if this method is executed
/// on non-AVX512 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xany_avx512_nofma_max_vertical(matrix: &[&[f64]]) -> Vec<f64> {
    let len = matrix[0].len();

    let mut max_values = vec![0.0; len];
    let max_values_ptr = max_values.as_mut_ptr();
    let offset_from = len % 64;

    // We work our way horizontally by taking steps of 64 and finding
    // the max of for each of the lanes vertically through the matrix.
    // TODO: I am unsure how hard this is on the cache or if there is a smarter
    //       way to write this.
    let mut i = 0;
    while i < (len - offset_from) {
        let mut acc1 = _mm512_set1_pd(f64::NEG_INFINITY);
        let mut acc2 = _mm512_set1_pd(f64::NEG_INFINITY);
        let mut acc3 = _mm512_set1_pd(f64::NEG_INFINITY);
        let mut acc4 = _mm512_set1_pd(f64::NEG_INFINITY);
        let mut acc5 = _mm512_set1_pd(f64::NEG_INFINITY);
        let mut acc6 = _mm512_set1_pd(f64::NEG_INFINITY);
        let mut acc7 = _mm512_set1_pd(f64::NEG_INFINITY);
        let mut acc8 = _mm512_set1_pd(f64::NEG_INFINITY);

        // Vertical max of the 64 elements.
        for m in 0..matrix.len() {
            let arr = *matrix.get_unchecked(m);
            debug_assert_eq!(arr.len(), len);

            let arr = arr.as_ptr();
            max_by_x64_horizontal(
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

        let result = mem::transmute::<[__m512d; 8], [f64; 64]>(merged);
        ptr::copy_nonoverlapping(result.as_ptr(), max_values_ptr.add(i), result.len());

        i += 64;
    }

    while i < len {
        let n = len - i;

        let mut acc = _mm512_set1_pd(f64::NEG_INFINITY);

        for m in 0..matrix.len() {
            let arr = *matrix.get_unchecked(m);
            debug_assert_eq!(arr.len(), len);

            let arr = arr.as_ptr();
            let x = load_one_variable_size_avx512_pd(arr.add(i), n);
            acc = _mm512_max_pd(acc, x);
        }

        copy_masked_avx512_pd_register_to(max_values_ptr.add(i), acc, n);

        i += 8;
    }

    max_values
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn max_by_x64_horizontal(
    arr: *const f64,
    acc1: &mut __m512d,
    acc2: &mut __m512d,
    acc3: &mut __m512d,
    acc4: &mut __m512d,
    acc5: &mut __m512d,
    acc6: &mut __m512d,
    acc7: &mut __m512d,
    acc8: &mut __m512d,
) {
    let [x1, x2, x3, x4] = offsets_avx512_pd::<CHUNK_0>(arr);
    let [x5, x6, x7, x8] = offsets_avx512_pd::<CHUNK_1>(arr);

    let x1 = _mm512_loadu_pd(x1);
    let x2 = _mm512_loadu_pd(x2);
    let x3 = _mm512_loadu_pd(x3);
    let x4 = _mm512_loadu_pd(x4);
    let x5 = _mm512_loadu_pd(x5);
    let x6 = _mm512_loadu_pd(x6);
    let x7 = _mm512_loadu_pd(x7);
    let x8 = _mm512_loadu_pd(x8);

    *acc1 = _mm512_max_pd(*acc1, x1);
    *acc2 = _mm512_max_pd(*acc2, x2);
    *acc3 = _mm512_max_pd(*acc3, x3);
    *acc4 = _mm512_max_pd(*acc4, x4);
    *acc5 = _mm512_max_pd(*acc5, x5);
    *acc6 = _mm512_max_pd(*acc6, x6);
    *acc7 = _mm512_max_pd(*acc7, x7);
    *acc8 = _mm512_max_pd(*acc8, x8);
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn rollup_max_acc(
    mut acc1: __m512d,
    acc2: __m512d,
    mut acc3: __m512d,
    acc4: __m512d,
    mut acc5: __m512d,
    acc6: __m512d,
    mut acc7: __m512d,
    acc8: __m512d,
) -> f64 {
    acc1 = _mm512_max_pd(acc1, acc2);
    acc3 = _mm512_max_pd(acc3, acc4);
    acc5 = _mm512_max_pd(acc5, acc6);
    acc7 = _mm512_max_pd(acc7, acc8);

    acc1 = _mm512_max_pd(acc1, acc3);
    acc5 = _mm512_max_pd(acc5, acc7);

    acc1 = _mm512_max_pd(acc1, acc5);

    _mm512_reduce_max_pd(acc1)
}

#[cfg(all(test, target_feature = "avx512f"))]
mod tests {
    use super::*;
    use crate::test_utils::get_sample_vectors;

    #[test]
    fn test_xconst_nofma_max_horizontal() {
        let (x, _) = get_sample_vectors(512);
        let max = unsafe { f64_xconst_avx512_nofma_max_horizontal::<512>(&x) };
        assert_eq!(max, x.iter().fold(f64::NEG_INFINITY, |acc, v| acc.max(*v)));
    }

    #[test]
    fn test_xconst_nofma_max_vertical() {
        let mut matrix = Vec::new();
        for _ in 0..25 {
            let (x, _) = get_sample_vectors(512);
            matrix.push(x);
        }

        let matrix_view = matrix.iter().map(|v| v.as_ref()).collect::<Vec<&[f64]>>();

        let mut expected_vertical_max = vec![f64::NEG_INFINITY; 512];
        for i in 0..512 {
            let mut max = f64::NEG_INFINITY;
            for arr in matrix.iter() {
                max = max.max(arr[i]);
            }
            expected_vertical_max[i] = max;
        }

        let max = unsafe { f64_xconst_avx512_nofma_max_vertical::<512>(&matrix_view) };
        assert_eq!(max, expected_vertical_max);
    }

    #[test]
    fn test_xany_nofma_max_horizontal() {
        let (x, _) = get_sample_vectors(537);
        let max = unsafe { f64_xany_avx512_nofma_max_horizontal(&x) };
        assert_eq!(max, x.iter().fold(f64::NEG_INFINITY, |acc, v| acc.max(*v)));
    }

    #[test]
    fn test_xany_nofma_max_vertical() {
        let mut matrix = Vec::new();
        for _ in 0..25 {
            let (x, _) = get_sample_vectors(537);
            matrix.push(x);
        }

        let matrix_view = matrix.iter().map(|v| v.as_ref()).collect::<Vec<&[f64]>>();

        let mut expected_vertical_max = vec![f64::NEG_INFINITY; 537];
        for i in 0..537 {
            let mut max = f64::NEG_INFINITY;
            for arr in matrix.iter() {
                max = max.max(arr[i]);
            }
            expected_vertical_max[i] = max;
        }

        let max = unsafe { f64_xany_avx512_nofma_max_vertical(&matrix_view) };
        assert_eq!(max, expected_vertical_max);
    }
}
