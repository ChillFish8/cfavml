use std::arch::x86_64::*;
use std::{mem, ptr};

use crate::danger::{
    f32_xany_avx2_fma_norm,
    f32_xany_avx2_nofma_norm,
    f32_xconst_avx2_fma_norm,
    f32_xconst_avx2_nofma_norm,
    offsets_avx2_ps,
    CHUNK_0,
    CHUNK_1,
};
use crate::math::*;

#[target_feature(enable = "avx2")]
#[inline]
/// Computes the angular hyperplane of two `[f32; DIMS]` vectors.
///
/// # Safety
///
/// DIMS **MUST** be a multiple of `64` and both vectors must be `DIMS` in length,
/// otherwise this routine will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_xconst_avx2_nofma_angular_hyperplane<const DIMS: usize>(
    x: &[f32],
    y: &[f32],
) -> Vec<f32> {
    debug_assert_eq!(DIMS % 64, 0);
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), DIMS);

    let mut norm_x = f32_xconst_avx2_nofma_norm::<DIMS>(x).sqrt();
    let mut norm_y = f32_xconst_avx2_nofma_norm::<DIMS>(y).sqrt();

    if norm_x.abs() < f32::EPSILON {
        norm_x = 1.0;
    }

    if norm_y.abs() < f32::EPSILON {
        norm_y = 1.0;
    }

    let mut hyperplane = vec![0.0; DIMS];

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();
    let hyperplane_ptr = hyperplane.as_mut_ptr();

    // Convert the norms to the inverse so we can use mul instructions
    // instead of divide operations. This prevents the system from
    // grinding to a crawl.
    let inverse_norm_x = _mm256_set1_ps(AutoMath::div(1.0, norm_x));
    let inverse_norm_y = _mm256_set1_ps(AutoMath::div(1.0, norm_y));

    let mut i = 0;
    while i < DIMS {
        let results = execute_f32_x64_block_normal_vector(
            x_ptr.add(i),
            y_ptr.add(i),
            inverse_norm_x,
            inverse_norm_y,
        );

        ptr::copy_nonoverlapping(results.as_ptr(), hyperplane_ptr.add(i), results.len());

        i += 64;
    }

    let mut norm_hyperplane = f32_xconst_avx2_nofma_norm::<DIMS>(&hyperplane).sqrt();
    if norm_hyperplane.abs() < f32::EPSILON {
        norm_hyperplane = 1.0;
    }

    // Convert the norms to the inverse so we can use mul instructions
    // instead of divide operations.This prevents the system from
    // grinding to a crawl.
    let inverse_norm_hyperplane = _mm256_set1_ps(AutoMath::div(1.0, norm_hyperplane));

    let mut i = 0;
    while i < DIMS {
        let results = execute_f32_x64_block_apply_norm(
            hyperplane_ptr.add(i),
            inverse_norm_hyperplane,
        );

        ptr::copy_nonoverlapping(results.as_ptr(), hyperplane_ptr.add(i), results.len());

        i += 64;
    }

    hyperplane
}

#[target_feature(enable = "avx2")]
#[inline]
/// Computes the angular hyperplane of two `f32` vectors.
///
/// # Safety
///
/// Vectors **MUST** be the same length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_xany_avx2_nofma_angular_hyperplane(x: &[f32], y: &[f32]) -> Vec<f32> {
    debug_assert_eq!(x.len(), y.len());

    let len = x.len();
    let offset_from = len % 64;

    let mut norm_x = f32_xany_avx2_nofma_norm(x).sqrt();
    let mut norm_y = f32_xany_avx2_nofma_norm(y).sqrt();

    if norm_x.abs() < f32::EPSILON {
        norm_x = 1.0;
    }

    if norm_y.abs() < f32::EPSILON {
        norm_y = 1.0;
    }

    let mut hyperplane = vec![0.0; len];

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();
    let hyperplane_ptr = hyperplane.as_mut_ptr();

    // Convert the norms to the inverse so we can use mul instructions
    // instead of divide operations. This prevents the system from
    // grinding to a crawl.
    let inverse_norm_x = _mm256_set1_ps(AutoMath::div(1.0, norm_x));
    let inverse_norm_y = _mm256_set1_ps(AutoMath::div(1.0, norm_y));

    let mut i = 0;
    while i < (len - offset_from) {
        let results = execute_f32_x64_block_normal_vector(
            x_ptr.add(i),
            y_ptr.add(i),
            inverse_norm_x,
            inverse_norm_y,
        );

        ptr::copy_nonoverlapping(results.as_ptr(), hyperplane_ptr.add(i), results.len());

        i += 64;
    }

    if offset_from != 0 {
        linear_apply_normal_vector::<AutoMath>(
            &x,
            &y,
            i,
            len,
            &mut hyperplane,
            AutoMath::div(1.0, norm_x),
            AutoMath::div(1.0, norm_y),
        );
    }

    let mut norm_hyperplane = f32_xany_avx2_nofma_norm(&hyperplane).sqrt();
    if norm_hyperplane.abs() < f32::EPSILON {
        norm_hyperplane = 1.0;
    }

    // Convert the norms to the inverse so we can use mul instructions
    // instead of divide operations.This prevents the system from
    // grinding to a crawl.
    let inverse_norm_hyperplane = _mm256_set1_ps(AutoMath::div(1.0, norm_hyperplane));

    let offset_from = len % 64;
    let mut i = 0;
    while i < (len - offset_from) {
        let results = execute_f32_x64_block_apply_norm(
            hyperplane_ptr.add(i),
            inverse_norm_hyperplane,
        );
        ptr::copy_nonoverlapping(results.as_ptr(), hyperplane_ptr.add(i), results.len());

        i += 64;
    }

    let offset_from = len % 64;
    if offset_from != 0 {
        linear_apply_norm::<AutoMath>(
            &mut hyperplane,
            i,
            len,
            AutoMath::div(1.0, norm_hyperplane),
        );
    }

    hyperplane
}

#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
/// Computes the angular hyperplane of two `[f32; DIMS]` vectors.
///
/// # Safety
///
/// DIMS **MUST** be a multiple of `64` and both vectors must be `DIMS` in length,
/// otherwise this routine will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_xconst_avx2_fma_angular_hyperplane<const DIMS: usize>(
    x: &[f32],
    y: &[f32],
) -> Vec<f32> {
    debug_assert_eq!(DIMS % 64, 0);
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), DIMS);

    let mut norm_x = f32_xconst_avx2_fma_norm::<DIMS>(x).sqrt();
    let mut norm_y = f32_xconst_avx2_fma_norm::<DIMS>(y).sqrt();

    if norm_x.abs() < f32::EPSILON {
        norm_x = 1.0;
    }

    if norm_y.abs() < f32::EPSILON {
        norm_y = 1.0;
    }

    let mut hyperplane = vec![0.0; DIMS];

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();
    let hyperplane_ptr = hyperplane.as_mut_ptr();

    // Convert the norms to the inverse so we can use mul instructions
    // instead of divide operations. This prevents the system from
    // grinding to a crawl.
    let inverse_norm_x = _mm256_set1_ps(AutoMath::div(1.0, norm_x));
    let inverse_norm_y = _mm256_set1_ps(AutoMath::div(1.0, norm_y));

    let mut i = 0;
    while i < DIMS {
        let results = execute_f32_x64_block_normal_vector(
            x_ptr.add(i),
            y_ptr.add(i),
            inverse_norm_x,
            inverse_norm_y,
        );

        ptr::copy_nonoverlapping(results.as_ptr(), hyperplane_ptr.add(i), results.len());

        i += 64;
    }

    let mut norm_hyperplane = f32_xconst_avx2_fma_norm::<DIMS>(&hyperplane).sqrt();
    if norm_hyperplane.abs() < f32::EPSILON {
        norm_hyperplane = 1.0;
    }

    // Convert the norms to the inverse so we can use mul instructions
    // instead of divide operations.This prevents the system from
    // grinding to a crawl.
    let inverse_norm_hyperplane = _mm256_set1_ps(AutoMath::div(1.0, norm_hyperplane));

    let mut i = 0;
    while i < DIMS {
        let results = execute_f32_x64_block_apply_norm(
            hyperplane_ptr.add(i),
            inverse_norm_hyperplane,
        );

        ptr::copy_nonoverlapping(results.as_ptr(), hyperplane_ptr.add(i), results.len());

        i += 64;
    }

    hyperplane
}

#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
/// Computes the angular hyperplane of two `f32` vectors.
///
/// # Safety
///
/// Vectors **MUST** be the same length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_xany_avx2_fma_angular_hyperplane(x: &[f32], y: &[f32]) -> Vec<f32> {
    debug_assert_eq!(x.len(), y.len());

    let len = x.len();
    let offset_from = len % 64;

    let mut norm_x = f32_xany_avx2_fma_norm(x).sqrt();
    let mut norm_y = f32_xany_avx2_fma_norm(y).sqrt();

    if norm_x.abs() < f32::EPSILON {
        norm_x = 1.0;
    }

    if norm_y.abs() < f32::EPSILON {
        norm_y = 1.0;
    }

    let mut hyperplane = vec![0.0; len];

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();
    let hyperplane_ptr = hyperplane.as_mut_ptr();

    // Convert the norms to the inverse so we can use mul instructions
    // instead of divide operations. This prevents the system from
    // grinding to a crawl.
    let inverse_norm_x = _mm256_set1_ps(AutoMath::div(1.0, norm_x));
    let inverse_norm_y = _mm256_set1_ps(AutoMath::div(1.0, norm_y));

    let mut i = 0;
    while i < (len - offset_from) {
        let results = execute_f32_x64_block_normal_vector(
            x_ptr.add(i),
            y_ptr.add(i),
            inverse_norm_x,
            inverse_norm_y,
        );

        ptr::copy_nonoverlapping(results.as_ptr(), hyperplane_ptr.add(i), results.len());

        i += 64;
    }

    if offset_from != 0 {
        linear_apply_normal_vector::<AutoMath>(
            x,
            y,
            i,
            len,
            &mut hyperplane,
            AutoMath::div(1.0, norm_x),
            AutoMath::div(1.0, norm_y),
        );
    }

    let mut norm_hyperplane = f32_xany_avx2_fma_norm(&hyperplane).sqrt();
    if norm_hyperplane.abs() < f32::EPSILON {
        norm_hyperplane = 1.0;
    }

    // Convert the norms to the inverse so we can use mul instructions
    // instead of divide operations.This prevents the system from
    // grinding to a crawl.
    let inverse_norm_hyperplane = _mm256_set1_ps(AutoMath::div(1.0, norm_hyperplane));

    let mut i = 0;
    while i < (len - offset_from) {
        let results = execute_f32_x64_block_apply_norm(
            hyperplane_ptr.add(i),
            inverse_norm_hyperplane,
        );
        ptr::copy_nonoverlapping(results.as_ptr(), hyperplane_ptr.add(i), results.len());

        i += 64;
    }

    let offset_from = len % 64;
    if offset_from != 0 {
        linear_apply_norm::<AutoMath>(
            &mut hyperplane,
            i,
            len,
            AutoMath::div(1.0, norm_hyperplane),
        );
    }

    hyperplane
}

#[inline]
unsafe fn linear_apply_normal_vector<M: Math<f32>>(
    x: &[f32],
    y: &[f32],
    start: usize,
    stop: usize,
    hyperplane: &mut [f32],
    inverse_norm_x: f32,
    inverse_norm_y: f32,
) {
    for i in start..stop {
        let x = *x.get_unchecked(i);
        let y = *y.get_unchecked(i);

        let norm_applied_x = M::mul(x, inverse_norm_x);
        let norm_applied_y = M::mul(y, inverse_norm_y);

        hyperplane[i] = M::sub(norm_applied_x, norm_applied_y);
    }
}

#[inline]
unsafe fn linear_apply_norm<M: Math<f32>>(
    hyperplane: &mut [f32],
    start: usize,
    stop: usize,
    inverse_norm_hyperplane: f32,
) {
    for i in start..stop {
        let x = hyperplane.get_unchecked_mut(i);
        *x = M::mul(*x, inverse_norm_hyperplane);
    }
}

#[inline(always)]
unsafe fn execute_f32_x64_block_normal_vector(
    x: *const f32,
    y: *const f32,
    inverse_norm_x: __m256,
    inverse_norm_y: __m256,
) -> [f32; 64] {
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

    let normalized_x1 = _mm256_mul_ps(x1, inverse_norm_x);
    let normalized_x2 = _mm256_mul_ps(x2, inverse_norm_x);
    let normalized_x3 = _mm256_mul_ps(x3, inverse_norm_x);
    let normalized_x4 = _mm256_mul_ps(x4, inverse_norm_x);
    let normalized_x5 = _mm256_mul_ps(x5, inverse_norm_x);
    let normalized_x6 = _mm256_mul_ps(x6, inverse_norm_x);
    let normalized_x7 = _mm256_mul_ps(x7, inverse_norm_x);
    let normalized_x8 = _mm256_mul_ps(x8, inverse_norm_x);

    let normalized_y1 = _mm256_mul_ps(y1, inverse_norm_y);
    let normalized_y2 = _mm256_mul_ps(y2, inverse_norm_y);
    let normalized_y3 = _mm256_mul_ps(y3, inverse_norm_y);
    let normalized_y4 = _mm256_mul_ps(y4, inverse_norm_y);
    let normalized_y5 = _mm256_mul_ps(y5, inverse_norm_y);
    let normalized_y6 = _mm256_mul_ps(y6, inverse_norm_y);
    let normalized_y7 = _mm256_mul_ps(y7, inverse_norm_y);
    let normalized_y8 = _mm256_mul_ps(y8, inverse_norm_y);

    let diff1 = _mm256_sub_ps(normalized_x1, normalized_y1);
    let diff2 = _mm256_sub_ps(normalized_x2, normalized_y2);
    let diff3 = _mm256_sub_ps(normalized_x3, normalized_y3);
    let diff4 = _mm256_sub_ps(normalized_x4, normalized_y4);
    let diff5 = _mm256_sub_ps(normalized_x5, normalized_y5);
    let diff6 = _mm256_sub_ps(normalized_x6, normalized_y6);
    let diff7 = _mm256_sub_ps(normalized_x7, normalized_y7);
    let diff8 = _mm256_sub_ps(normalized_x8, normalized_y8);

    let lanes = [diff1, diff2, diff3, diff4, diff5, diff6, diff7, diff8];

    mem::transmute(lanes)
}

#[inline(always)]
unsafe fn execute_f32_x64_block_apply_norm(
    x: *const f32,
    inverse_norm_x: __m256,
) -> [f32; 64] {
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

    let normalized_x1 = _mm256_mul_ps(x1, inverse_norm_x);
    let normalized_x2 = _mm256_mul_ps(x2, inverse_norm_x);
    let normalized_x3 = _mm256_mul_ps(x3, inverse_norm_x);
    let normalized_x4 = _mm256_mul_ps(x4, inverse_norm_x);
    let normalized_x5 = _mm256_mul_ps(x5, inverse_norm_x);
    let normalized_x6 = _mm256_mul_ps(x6, inverse_norm_x);
    let normalized_x7 = _mm256_mul_ps(x7, inverse_norm_x);
    let normalized_x8 = _mm256_mul_ps(x8, inverse_norm_x);

    let lanes = [
        normalized_x1,
        normalized_x2,
        normalized_x3,
        normalized_x4,
        normalized_x5,
        normalized_x6,
        normalized_x7,
        normalized_x8,
    ];

    mem::transmute(lanes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{
        assert_is_close_vector,
        get_sample_vectors,
        simple_angular_hyperplane,
    };

    #[cfg(feature = "nightly")]
    #[test]
    fn test_xconst_fma_angular_hyperplane() {
        let (x, y) = get_sample_vectors(1024);
        let hyperplane =
            unsafe { f32_xconst_avx2_fma_angular_hyperplane::<1024>(&x, &y) };
        let expected = simple_angular_hyperplane(&x, &y);
        assert_is_close_vector(&hyperplane, &expected);
    }

    #[test]
    fn test_xconst_nofma_angular_hyperplane() {
        let (x, y) = get_sample_vectors(1024);
        let hyperplane =
            unsafe { f32_xconst_avx2_nofma_angular_hyperplane::<1024>(&x, &y) };
        let expected = simple_angular_hyperplane(&x, &y);
        assert_is_close_vector(&hyperplane, &expected);
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_xany_fma_angular_hyperplane() {
        let (x, y) = get_sample_vectors(127);
        let hyperplane = unsafe { f32_xany_avx2_fma_angular_hyperplane(&x, &y) };
        let expected = simple_angular_hyperplane(&x, &y);
        assert_is_close_vector(&hyperplane, &expected);
    }

    #[test]
    fn test_xany_nofma_angular_hyperplane() {
        let (x, y) = get_sample_vectors(127);
        let hyperplane = unsafe { f32_xany_avx2_nofma_angular_hyperplane(&x, &y) };
        let expected = simple_angular_hyperplane(&x, &y);
        assert_is_close_vector(&hyperplane, &expected);
    }
}
