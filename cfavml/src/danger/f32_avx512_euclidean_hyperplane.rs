use std::arch::x86_64::*;
use std::{mem, ptr};

use crate::danger::{
    copy_masked_avx512_ps_register_to,
    load_two_variable_size_avx512_ps,
    offsets_avx512_ps,
    sum_avx512_x8_ps,
    CHUNK_0,
    CHUNK_1,
};

#[target_feature(enable = "avx512f")]
#[inline]
/// Computes the Euclidean hyperplane of two `[f32; DIMS]` vectors
/// and the offset from origin.
///
/// # Safety
///
/// DIMS **MUST** be a multiple of `128` and vectors must be `DIMS` in length,
/// otherwise this routine will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_xconst_avx512_fma_euclidean_hyperplane<const DIMS: usize>(
    x: &[f32],
    y: &[f32],
) -> (Vec<f32>, f32) {
    debug_assert_eq!(DIMS % 128, 0);
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), DIMS);

    let mut hyperplane = vec![0.0; DIMS];
    let hyperplane_ptr = hyperplane.as_mut_ptr();

    let x = x.as_ptr();
    let y = y.as_ptr();

    let mut offset_acc1 = _mm512_setzero_ps();
    let mut offset_acc2 = _mm512_setzero_ps();
    let mut offset_acc3 = _mm512_setzero_ps();
    let mut offset_acc4 = _mm512_setzero_ps();
    let mut offset_acc5 = _mm512_setzero_ps();
    let mut offset_acc6 = _mm512_setzero_ps();
    let mut offset_acc7 = _mm512_setzero_ps();
    let mut offset_acc8 = _mm512_setzero_ps();

    let mut i = 0;
    while i < DIMS {
        let results = execute_f32_x128_block_fma_hyperplane(
            x.add(i),
            y.add(i),
            &mut offset_acc1,
            &mut offset_acc2,
            &mut offset_acc3,
            &mut offset_acc4,
            &mut offset_acc5,
            &mut offset_acc6,
            &mut offset_acc7,
            &mut offset_acc8,
        );

        ptr::copy_nonoverlapping(results.as_ptr(), hyperplane_ptr.add(i), results.len());

        i += 128;
    }

    let hyperplane_offset = -sum_avx512_x8_ps(
        offset_acc1,
        offset_acc2,
        offset_acc3,
        offset_acc4,
        offset_acc5,
        offset_acc6,
        offset_acc7,
        offset_acc8,
    );

    (hyperplane, hyperplane_offset)
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Computes the Euclidean hyperplane of two `f32` vectors
/// and the offset from origin.
///
/// # Safety
///
/// Vectors **MUST** be equal length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_xany_avx512_fma_euclidean_hyperplane(
    x: &[f32],
    y: &[f32],
) -> (Vec<f32>, f32) {
    debug_assert_eq!(x.len(), y.len());

    let len = x.len();
    let offset_from = len % 128;
    let mut hyperplane = vec![0.0; len];
    let hyperplane_ptr = hyperplane.as_mut_ptr();

    let x = x.as_ptr();
    let y = y.as_ptr();

    let div_by_2 = _mm512_set1_ps(0.5);
    let mut offset_acc1 = _mm512_setzero_ps();
    let mut offset_acc2 = _mm512_setzero_ps();
    let mut offset_acc3 = _mm512_setzero_ps();
    let mut offset_acc4 = _mm512_setzero_ps();
    let mut offset_acc5 = _mm512_setzero_ps();
    let mut offset_acc6 = _mm512_setzero_ps();
    let mut offset_acc7 = _mm512_setzero_ps();
    let mut offset_acc8 = _mm512_setzero_ps();

    let mut i = 0;
    while i < (len - offset_from) {
        let results = execute_f32_x128_block_fma_hyperplane(
            x.add(i),
            y.add(i),
            &mut offset_acc1,
            &mut offset_acc2,
            &mut offset_acc3,
            &mut offset_acc4,
            &mut offset_acc5,
            &mut offset_acc6,
            &mut offset_acc7,
            &mut offset_acc8,
        );

        ptr::copy_nonoverlapping(results.as_ptr(), hyperplane_ptr.add(i), results.len());

        i += 128;
    }

    while i < len {
        let n = len - i;
        let (x, y) = load_two_variable_size_avx512_ps(x.add(i), y.add(i), n);

        let diff = _mm512_sub_ps(x, y);
        let sum = _mm512_add_ps(x, y);
        let mean = _mm512_mul_ps(sum, div_by_2);

        offset_acc1 = _mm512_fmadd_ps(diff, mean, offset_acc1);

        copy_masked_avx512_ps_register_to(hyperplane_ptr.add(i), diff, n);

        i += 16;
    }

    let hyperplane_offset = -sum_avx512_x8_ps(
        offset_acc1,
        offset_acc2,
        offset_acc3,
        offset_acc4,
        offset_acc5,
        offset_acc6,
        offset_acc7,
        offset_acc8,
    );

    (hyperplane, hyperplane_offset)
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn execute_f32_x128_block_fma_hyperplane(
    x: *const f32,
    y: *const f32,
    offset_acc1: &mut __m512,
    offset_acc2: &mut __m512,
    offset_acc3: &mut __m512,
    offset_acc4: &mut __m512,
    offset_acc5: &mut __m512,
    offset_acc6: &mut __m512,
    offset_acc7: &mut __m512,
    offset_acc8: &mut __m512,
) -> [f32; 128] {
    // TODO: Hopefully LLVM is smart enough to optimize this out, but we should
    //       double check that we don't reset the register each time.
    let div_by_2 = _mm512_set1_ps(0.5);

    let [x1, x2, x3, x4] = offsets_avx512_ps::<CHUNK_0>(x);
    let [x5, x6, x7, x8] = offsets_avx512_ps::<CHUNK_1>(x);

    let [y1, y2, y3, y4] = offsets_avx512_ps::<CHUNK_0>(y);
    let [y5, y6, y7, y8] = offsets_avx512_ps::<CHUNK_1>(y);

    let x1 = _mm512_loadu_ps(x1);
    let x2 = _mm512_loadu_ps(x2);
    let x3 = _mm512_loadu_ps(x3);
    let x4 = _mm512_loadu_ps(x4);
    let x5 = _mm512_loadu_ps(x5);
    let x6 = _mm512_loadu_ps(x6);
    let x7 = _mm512_loadu_ps(x7);
    let x8 = _mm512_loadu_ps(x8);

    let y1 = _mm512_loadu_ps(y1);
    let y2 = _mm512_loadu_ps(y2);
    let y3 = _mm512_loadu_ps(y3);
    let y4 = _mm512_loadu_ps(y4);
    let y5 = _mm512_loadu_ps(y5);
    let y6 = _mm512_loadu_ps(y6);
    let y7 = _mm512_loadu_ps(y7);
    let y8 = _mm512_loadu_ps(y8);

    let diff1 = _mm512_sub_ps(x1, y1);
    let diff2 = _mm512_sub_ps(x2, y2);
    let diff3 = _mm512_sub_ps(x3, y3);
    let diff4 = _mm512_sub_ps(x4, y4);
    let diff5 = _mm512_sub_ps(x5, y5);
    let diff6 = _mm512_sub_ps(x6, y6);
    let diff7 = _mm512_sub_ps(x7, y7);
    let diff8 = _mm512_sub_ps(x8, y8);

    let sum1 = _mm512_add_ps(x1, y1);
    let sum2 = _mm512_add_ps(x2, y2);
    let sum3 = _mm512_add_ps(x3, y3);
    let sum4 = _mm512_add_ps(x4, y4);
    let sum5 = _mm512_add_ps(x5, y5);
    let sum6 = _mm512_add_ps(x6, y6);
    let sum7 = _mm512_add_ps(x7, y7);
    let sum8 = _mm512_add_ps(x8, y8);

    let mean1 = _mm512_mul_ps(sum1, div_by_2);
    let mean2 = _mm512_mul_ps(sum2, div_by_2);
    let mean3 = _mm512_mul_ps(sum3, div_by_2);
    let mean4 = _mm512_mul_ps(sum4, div_by_2);
    let mean5 = _mm512_mul_ps(sum5, div_by_2);
    let mean6 = _mm512_mul_ps(sum6, div_by_2);
    let mean7 = _mm512_mul_ps(sum7, div_by_2);
    let mean8 = _mm512_mul_ps(sum8, div_by_2);

    *offset_acc1 = _mm512_fmadd_ps(diff1, mean1, *offset_acc1);
    *offset_acc2 = _mm512_fmadd_ps(diff2, mean2, *offset_acc2);
    *offset_acc3 = _mm512_fmadd_ps(diff3, mean3, *offset_acc3);
    *offset_acc4 = _mm512_fmadd_ps(diff4, mean4, *offset_acc4);
    *offset_acc5 = _mm512_fmadd_ps(diff5, mean5, *offset_acc5);
    *offset_acc6 = _mm512_fmadd_ps(diff6, mean6, *offset_acc6);
    *offset_acc7 = _mm512_fmadd_ps(diff7, mean7, *offset_acc7);
    *offset_acc8 = _mm512_fmadd_ps(diff8, mean8, *offset_acc8);

    let plane = [diff1, diff2, diff3, diff4, diff5, diff6, diff7, diff8];

    mem::transmute(plane)
}

#[cfg(all(test, target_feature = "avx512f"))]
mod tests {
    use super::*;
    use crate::test_utils::{
        assert_is_close,
        assert_is_close_vector,
        get_sample_vectors,
        simple_euclidean_hyperplane,
    };

    #[test]
    fn test_xconst_fma_euclidean_hyperplane() {
        let (x, y) = get_sample_vectors(1024);
        let (hyperplane, offset) =
            unsafe { f32_xconst_avx512_fma_euclidean_hyperplane::<1024>(&x, &y) };
        let (expected, expected_offset) = simple_euclidean_hyperplane(&x, &y);
        assert_is_close(offset, expected_offset);
        assert_is_close_vector(&hyperplane, &expected);
    }

    #[test]
    fn test_xany_fma_euclidean_hyperplane() {
        let (x, y) = get_sample_vectors(563);
        let (hyperplane, offset) =
            unsafe { f32_xany_avx512_fma_euclidean_hyperplane(&x, &y) };
        let (expected, expected_offset) = simple_euclidean_hyperplane(&x, &y);
        assert_is_close(offset, expected_offset);
        assert_is_close_vector(&hyperplane, &expected);
    }
}
