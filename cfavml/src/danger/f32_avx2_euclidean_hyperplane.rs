use std::arch::x86_64::*;
use std::{mem, ptr};

use crate::danger::{
    fallback_euclidean_hyperplane,
    offsets_avx2_ps,
    rollup_x8_ps,
    sum_avx2_ps,
    CHUNK_0,
    CHUNK_1,
};
use crate::math::*;

#[target_feature(enable = "avx2")]
#[inline]
/// Computes the Euclidean hyperplane of two `[f32; DIMS]` vectors.
///
/// # Safety
///
/// DIMS **MUST** be a multiple of `64` and both vectors must be `DIMS` in length,
/// otherwise this routine will become immediately UB due to out of bounds pointer accesses.
///
/// The lengths of `x` and `y` **must** match and contain only finite values.
pub unsafe fn f32_xconst_avx2_nofma_euclidean_hyperplane<const DIMS: usize>(
    x: &[f32],
    y: &[f32],
) -> (Vec<f32>, f32) {
    debug_assert_eq!(DIMS % 64, 0);
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), DIMS);

    let mut hyperplane = vec![0.0; DIMS];

    let x = x.as_ptr();
    let y = y.as_ptr();

    let mut offset_acc1 = _mm256_setzero_ps();
    let mut offset_acc2 = _mm256_setzero_ps();
    let mut offset_acc3 = _mm256_setzero_ps();
    let mut offset_acc4 = _mm256_setzero_ps();
    let mut offset_acc5 = _mm256_setzero_ps();
    let mut offset_acc6 = _mm256_setzero_ps();
    let mut offset_acc7 = _mm256_setzero_ps();
    let mut offset_acc8 = _mm256_setzero_ps();

    let mut i = 0;
    while i < DIMS {
        let results = execute_f32_x64_block_nofma_hyperplane(
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

        ptr::copy_nonoverlapping(
            results.as_ptr(),
            hyperplane.as_mut_ptr().add(i),
            results.len(),
        );

        i += 64;
    }

    let hyperplane_offset = sub_reduce_x8(
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

#[target_feature(enable = "avx2")]
#[inline]
/// Computes the Euclidean hyperplane of two f32 vectors of any size, assuming
/// the size of `x` and `y` are the same size.
///
/// # Safety
///
/// The lengths of `x` and `y` **must** match and contain only finite values.
pub unsafe fn f32_xany_avx2_nofma_euclidean_hyperplane(
    x: &[f32],
    y: &[f32],
) -> (Vec<f32>, f32) {
    debug_assert_eq!(x.len(), y.len(), "Provided vectors must match in size");

    let len = x.len();
    let offset_from = len % 64;

    let mut hyperplane_offset = 0.0;
    let mut hyperplane = vec![0.0; len];

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();

    let mut offset_acc1 = _mm256_setzero_ps();
    let mut offset_acc2 = _mm256_setzero_ps();
    let mut offset_acc3 = _mm256_setzero_ps();
    let mut offset_acc4 = _mm256_setzero_ps();
    let mut offset_acc5 = _mm256_setzero_ps();
    let mut offset_acc6 = _mm256_setzero_ps();
    let mut offset_acc7 = _mm256_setzero_ps();
    let mut offset_acc8 = _mm256_setzero_ps();

    let mut i = 0;
    while i < (len - offset_from) {
        let results = execute_f32_x64_block_nofma_hyperplane(
            x_ptr.add(i),
            y_ptr.add(i),
            &mut offset_acc1,
            &mut offset_acc2,
            &mut offset_acc3,
            &mut offset_acc4,
            &mut offset_acc5,
            &mut offset_acc6,
            &mut offset_acc7,
            &mut offset_acc8,
        );

        ptr::copy_nonoverlapping(
            results.as_ptr(),
            hyperplane.as_mut_ptr().add(i),
            results.len(),
        );

        i += 64;
    }

    if offset_from != 0 {
        let x_subsection = &x[(len - offset_from)..];
        let y_subsection = &y[(len - offset_from)..];
        hyperplane_offset = fallback_euclidean_hyperplane::<AutoMath>(
            x_subsection,
            y_subsection,
            &mut hyperplane[(len - offset_from)..],
        );
    }

    hyperplane_offset += sub_reduce_x8(
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

#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
/// Computes the Euclidean hyperplane of two `[f32; DIMS]` vectors.
///
/// # Safety
///
/// DIMS **MUST** be a multiple of `64` and both vectors must be `DIMS` in length,
/// otherwise this routine will become immediately UB due to out of bounds pointer accesses.
///
/// The lengths of `x` and `y` **must** match and contain only finite values.
pub unsafe fn f32_xconst_avx2_fma_euclidean_hyperplane<const DIMS: usize>(
    x: &[f32],
    y: &[f32],
) -> (Vec<f32>, f32) {
    debug_assert_eq!(DIMS % 64, 0);
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), DIMS);

    let mut hyperplane = vec![0.0; DIMS];

    let x = x.as_ptr();
    let y = y.as_ptr();

    let mut offset_acc1 = _mm256_setzero_ps();
    let mut offset_acc2 = _mm256_setzero_ps();
    let mut offset_acc3 = _mm256_setzero_ps();
    let mut offset_acc4 = _mm256_setzero_ps();
    let mut offset_acc5 = _mm256_setzero_ps();
    let mut offset_acc6 = _mm256_setzero_ps();
    let mut offset_acc7 = _mm256_setzero_ps();
    let mut offset_acc8 = _mm256_setzero_ps();

    let mut i = 0;
    while i < DIMS {
        let results = execute_f32_x64_block_fma_hyperplane(
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

        ptr::copy_nonoverlapping(
            results.as_ptr(),
            hyperplane.as_mut_ptr().add(i),
            results.len(),
        );

        i += 64;
    }

    let hyperplane_offset = sub_reduce_x8(
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

#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
/// Computes the Euclidean hyperplane of two f32 vectors of any size, assuming
/// the size of `x` and `y` are the same size.
///
/// # Safety
///
/// The lengths of `x` and `y` **must** match and contain only finite values.
pub unsafe fn f32_xany_avx2_fma_euclidean_hyperplane(
    x: &[f32],
    y: &[f32],
) -> (Vec<f32>, f32) {
    debug_assert_eq!(x.len(), y.len(), "Provided vectors must match in size");

    let len = x.len();
    let offset_from = len % 64;
    let mut hyperplane_offset = 0.0;
    let mut hyperplane = vec![0.0; len];

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();

    let mut offset_acc1 = _mm256_setzero_ps();
    let mut offset_acc2 = _mm256_setzero_ps();
    let mut offset_acc3 = _mm256_setzero_ps();
    let mut offset_acc4 = _mm256_setzero_ps();
    let mut offset_acc5 = _mm256_setzero_ps();
    let mut offset_acc6 = _mm256_setzero_ps();
    let mut offset_acc7 = _mm256_setzero_ps();
    let mut offset_acc8 = _mm256_setzero_ps();

    let mut i = 0;
    while i < (len - offset_from) {
        let results = execute_f32_x64_block_fma_hyperplane(
            x_ptr.add(i),
            y_ptr.add(i),
            &mut offset_acc1,
            &mut offset_acc2,
            &mut offset_acc3,
            &mut offset_acc4,
            &mut offset_acc5,
            &mut offset_acc6,
            &mut offset_acc7,
            &mut offset_acc8,
        );

        ptr::copy_nonoverlapping(
            results.as_ptr(),
            hyperplane.as_mut_ptr().add(i),
            results.len(),
        );

        i += 64;
    }

    if offset_from != 0 {
        let x_subsection = &x[(len - offset_from)..];
        let y_subsection = &y[(len - offset_from)..];
        hyperplane_offset = fallback_euclidean_hyperplane::<AutoMath>(
            x_subsection,
            y_subsection,
            &mut hyperplane[(len - offset_from)..],
        );
    }

    hyperplane_offset = AutoMath::add(
        hyperplane_offset,
        sub_reduce_x8(
            offset_acc1,
            offset_acc2,
            offset_acc3,
            offset_acc4,
            offset_acc5,
            offset_acc6,
            offset_acc7,
            offset_acc8,
        ),
    );

    (hyperplane, hyperplane_offset)
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn execute_f32_x64_block_nofma_hyperplane(
    x: *const f32,
    y: *const f32,
    offset_acc1: &mut __m256,
    offset_acc2: &mut __m256,
    offset_acc3: &mut __m256,
    offset_acc4: &mut __m256,
    offset_acc5: &mut __m256,
    offset_acc6: &mut __m256,
    offset_acc7: &mut __m256,
    offset_acc8: &mut __m256,
) -> [f32; 64] {
    // TODO: Hopefully LLVM is smart enough to optimize this out, but we should
    //       double check that we don't reset the register each time.
    let div_by_2 = _mm256_set1_ps(0.5);

    let [x1, x2, x3, x4] = offsets_avx2_ps::<CHUNK_0>(x);
    let [x5, x6, x7, x8] = offsets_avx2_ps::<CHUNK_1>(x);

    let [y1, y2, y3, y4] = offsets_avx2_ps::<CHUNK_0>(y);
    let [y5, y6, y7, y8] = offsets_avx2_ps::<CHUNK_1>(y);

    let x1 = _mm256_loadu_ps(x1);
    let x2 = _mm256_loadu_ps(x2);
    let x3 = _mm256_loadu_ps(x3);
    let x4 = _mm256_loadu_ps(x4);
    let x5 = _mm256_loadu_ps(x5);
    let x6 = _mm256_loadu_ps(x6);
    let x7 = _mm256_loadu_ps(x7);
    let x8 = _mm256_loadu_ps(x8);

    let y1 = _mm256_loadu_ps(y1);
    let y2 = _mm256_loadu_ps(y2);
    let y3 = _mm256_loadu_ps(y3);
    let y4 = _mm256_loadu_ps(y4);
    let y5 = _mm256_loadu_ps(y5);
    let y6 = _mm256_loadu_ps(y6);
    let y7 = _mm256_loadu_ps(y7);
    let y8 = _mm256_loadu_ps(y8);

    let diff1 = _mm256_sub_ps(x1, y1);
    let diff2 = _mm256_sub_ps(x2, y2);
    let diff3 = _mm256_sub_ps(x3, y3);
    let diff4 = _mm256_sub_ps(x4, y4);
    let diff5 = _mm256_sub_ps(x5, y5);
    let diff6 = _mm256_sub_ps(x6, y6);
    let diff7 = _mm256_sub_ps(x7, y7);
    let diff8 = _mm256_sub_ps(x8, y8);

    let sum1 = _mm256_add_ps(x1, y1);
    let sum2 = _mm256_add_ps(x2, y2);
    let sum3 = _mm256_add_ps(x3, y3);
    let sum4 = _mm256_add_ps(x4, y4);
    let sum5 = _mm256_add_ps(x5, y5);
    let sum6 = _mm256_add_ps(x6, y6);
    let sum7 = _mm256_add_ps(x7, y7);
    let sum8 = _mm256_add_ps(x8, y8);

    let mean1 = _mm256_mul_ps(sum1, div_by_2);
    let mean2 = _mm256_mul_ps(sum2, div_by_2);
    let mean3 = _mm256_mul_ps(sum3, div_by_2);
    let mean4 = _mm256_mul_ps(sum4, div_by_2);
    let mean5 = _mm256_mul_ps(sum5, div_by_2);
    let mean6 = _mm256_mul_ps(sum6, div_by_2);
    let mean7 = _mm256_mul_ps(sum7, div_by_2);
    let mean8 = _mm256_mul_ps(sum8, div_by_2);

    let r1 = _mm256_mul_ps(diff1, mean1);
    let r2 = _mm256_mul_ps(diff2, mean2);
    let r3 = _mm256_mul_ps(diff3, mean3);
    let r4 = _mm256_mul_ps(diff4, mean4);
    let r5 = _mm256_mul_ps(diff5, mean5);
    let r6 = _mm256_mul_ps(diff6, mean6);
    let r7 = _mm256_mul_ps(diff7, mean7);
    let r8 = _mm256_mul_ps(diff8, mean8);

    *offset_acc1 = _mm256_add_ps(*offset_acc1, r1);
    *offset_acc2 = _mm256_add_ps(*offset_acc2, r2);
    *offset_acc3 = _mm256_add_ps(*offset_acc3, r3);
    *offset_acc4 = _mm256_add_ps(*offset_acc4, r4);
    *offset_acc5 = _mm256_add_ps(*offset_acc5, r5);
    *offset_acc6 = _mm256_add_ps(*offset_acc6, r6);
    *offset_acc7 = _mm256_add_ps(*offset_acc7, r7);
    *offset_acc8 = _mm256_add_ps(*offset_acc8, r8);

    let plane = [diff1, diff2, diff3, diff4, diff5, diff6, diff7, diff8];

    mem::transmute(plane)
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn execute_f32_x64_block_fma_hyperplane(
    x: *const f32,
    y: *const f32,
    offset_acc1: &mut __m256,
    offset_acc2: &mut __m256,
    offset_acc3: &mut __m256,
    offset_acc4: &mut __m256,
    offset_acc5: &mut __m256,
    offset_acc6: &mut __m256,
    offset_acc7: &mut __m256,
    offset_acc8: &mut __m256,
) -> [f32; 64] {
    // TODO: Hopefully LLVM is smart enough to optimize this out, but we should
    //       double check that we don't reset the register each time.
    let div_by_2 = _mm256_set1_ps(0.5);

    let [x1, x2, x3, x4] = offsets_avx2_ps::<CHUNK_0>(x);
    let [x5, x6, x7, x8] = offsets_avx2_ps::<CHUNK_1>(x);

    let [y1, y2, y3, y4] = offsets_avx2_ps::<CHUNK_0>(y);
    let [y5, y6, y7, y8] = offsets_avx2_ps::<CHUNK_1>(y);

    let x1 = _mm256_loadu_ps(x1);
    let x2 = _mm256_loadu_ps(x2);
    let x3 = _mm256_loadu_ps(x3);
    let x4 = _mm256_loadu_ps(x4);
    let x5 = _mm256_loadu_ps(x5);
    let x6 = _mm256_loadu_ps(x6);
    let x7 = _mm256_loadu_ps(x7);
    let x8 = _mm256_loadu_ps(x8);

    let y1 = _mm256_loadu_ps(y1);
    let y2 = _mm256_loadu_ps(y2);
    let y3 = _mm256_loadu_ps(y3);
    let y4 = _mm256_loadu_ps(y4);
    let y5 = _mm256_loadu_ps(y5);
    let y6 = _mm256_loadu_ps(y6);
    let y7 = _mm256_loadu_ps(y7);
    let y8 = _mm256_loadu_ps(y8);

    let diff1 = _mm256_sub_ps(x1, y1);
    let diff2 = _mm256_sub_ps(x2, y2);
    let diff3 = _mm256_sub_ps(x3, y3);
    let diff4 = _mm256_sub_ps(x4, y4);
    let diff5 = _mm256_sub_ps(x5, y5);
    let diff6 = _mm256_sub_ps(x6, y6);
    let diff7 = _mm256_sub_ps(x7, y7);
    let diff8 = _mm256_sub_ps(x8, y8);

    let sum1 = _mm256_add_ps(x1, y1);
    let sum2 = _mm256_add_ps(x2, y2);
    let sum3 = _mm256_add_ps(x3, y3);
    let sum4 = _mm256_add_ps(x4, y4);
    let sum5 = _mm256_add_ps(x5, y5);
    let sum6 = _mm256_add_ps(x6, y6);
    let sum7 = _mm256_add_ps(x7, y7);
    let sum8 = _mm256_add_ps(x8, y8);

    let mean1 = _mm256_mul_ps(sum1, div_by_2);
    let mean2 = _mm256_mul_ps(sum2, div_by_2);
    let mean3 = _mm256_mul_ps(sum3, div_by_2);
    let mean4 = _mm256_mul_ps(sum4, div_by_2);
    let mean5 = _mm256_mul_ps(sum5, div_by_2);
    let mean6 = _mm256_mul_ps(sum6, div_by_2);
    let mean7 = _mm256_mul_ps(sum7, div_by_2);
    let mean8 = _mm256_mul_ps(sum8, div_by_2);

    *offset_acc1 = _mm256_fmadd_ps(diff1, mean1, *offset_acc1);
    *offset_acc2 = _mm256_fmadd_ps(diff2, mean2, *offset_acc2);
    *offset_acc3 = _mm256_fmadd_ps(diff3, mean3, *offset_acc3);
    *offset_acc4 = _mm256_fmadd_ps(diff4, mean4, *offset_acc4);
    *offset_acc5 = _mm256_fmadd_ps(diff5, mean5, *offset_acc5);
    *offset_acc6 = _mm256_fmadd_ps(diff6, mean6, *offset_acc6);
    *offset_acc7 = _mm256_fmadd_ps(diff7, mean7, *offset_acc7);
    *offset_acc8 = _mm256_fmadd_ps(diff8, mean8, *offset_acc8);

    let plane = [diff1, diff2, diff3, diff4, diff5, diff6, diff7, diff8];

    mem::transmute(plane)
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn sub_reduce_x8(
    mut acc1: __m256,
    acc2: __m256,
    acc3: __m256,
    acc4: __m256,
    acc5: __m256,
    acc6: __m256,
    acc7: __m256,
    acc8: __m256,
) -> f32 {
    acc1 = rollup_x8_ps(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8);
    -sum_avx2_ps(acc1)
}

#[cfg(test)]
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
            unsafe { f32_xconst_avx2_fma_euclidean_hyperplane::<1024>(&x, &y) };
        let (expected, expected_offset) = simple_euclidean_hyperplane(&x, &y);
        assert_is_close(offset, expected_offset);
        assert_is_close_vector(&hyperplane, &expected);
    }

    #[test]
    fn test_xconst_nofma_euclidean_hyperplane() {
        let (x, y) = get_sample_vectors(1024);
        let (hyperplane, offset) =
            unsafe { f32_xconst_avx2_nofma_euclidean_hyperplane::<1024>(&x, &y) };
        let (expected, expected_offset) = simple_euclidean_hyperplane(&x, &y);
        assert_is_close(offset, expected_offset);
        assert_is_close_vector(&hyperplane, &expected);
    }

    #[test]
    fn test_xany_fma_euclidean_hyperplane() {
        let (x, y) = get_sample_vectors(127);
        let (hyperplane, offset) =
            unsafe { f32_xany_avx2_fma_euclidean_hyperplane(&x, &y) };
        let (expected, expected_offset) = simple_euclidean_hyperplane(&x, &y);
        assert_is_close(offset, expected_offset);
        assert_is_close_vector(&hyperplane, &expected);
    }

    #[test]
    fn test_xany_nofma_euclidean_hyperplane() {
        let (x, y) = get_sample_vectors(127);
        let (hyperplane, offset) =
            unsafe { f32_xany_avx2_nofma_euclidean_hyperplane(&x, &y) };
        let (expected, expected_offset) = simple_euclidean_hyperplane(&x, &y);
        assert_is_close(offset, expected_offset);
        assert_is_close_vector(&hyperplane, &expected);
    }
}
