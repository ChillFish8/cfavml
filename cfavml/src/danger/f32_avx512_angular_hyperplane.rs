use std::arch::x86_64::*;
use std::{mem, ptr};

use crate::danger::{
    copy_masked_avx512_ps_register_to,
    f32_xany_avx512_fma_norm,
    f32_xany_avx512_nofma_div_value,
    f32_xconst_avx512_fma_norm,
    f32_xconst_avx512_nofma_div_value,
    load_two_variable_size_avx512_ps,
    offsets_avx512_ps,
    CHUNK_0,
    CHUNK_1,
};

#[target_feature(enable = "avx512f")]
#[inline]
/// Computes the angular hyperplane of two `[f32; DIMS]` vectors.
///
/// # Safety
///
/// DIMS **MUST** be a multiple of `128` and vectors must be `DIMS` in length,
/// otherwise this routine will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_xconst_avx512_fma_angular_hyperplane<const DIMS: usize>(
    x: &[f32],
    y: &[f32],
) -> Vec<f32> {
    debug_assert_eq!(DIMS % 128, 0);
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), DIMS);

    let mut norm_x = f32_xconst_avx512_fma_norm::<DIMS>(x).sqrt();
    if norm_x.abs() < f32::EPSILON {
        norm_x = 1.0;
    }

    let mut norm_y = f32_xconst_avx512_fma_norm::<DIMS>(y).sqrt();
    if norm_y.abs() < f32::EPSILON {
        norm_y = 1.0;
    }

    let x = x.as_ptr();
    let y = y.as_ptr();

    let inverse_norm_x = _mm512_set1_ps(1.0 / norm_x);
    let inverse_norm_y = _mm512_set1_ps(1.0 / norm_y);

    let mut hyperplane = vec![0.0; DIMS];
    let hyperplane_ptr = hyperplane.as_mut_ptr();

    let mut i = 0;
    while i < DIMS {
        let results = execute_f32_x128_block_normal_vector(
            x.add(i),
            y.add(i),
            inverse_norm_x,
            inverse_norm_y,
        );
        ptr::copy_nonoverlapping(results.as_ptr(), hyperplane_ptr.add(i), results.len());

        i += 128;
    }

    let mut norm_hyperplane = f32_xconst_avx512_fma_norm::<DIMS>(&hyperplane).sqrt();
    if norm_hyperplane.abs() < f32::EPSILON {
        norm_hyperplane = 1.0;
    }

    f32_xconst_avx512_nofma_div_value::<DIMS>(&mut hyperplane, norm_hyperplane);

    hyperplane
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Computes the angular hyperplane of two `f32` vectors.
///
/// # Safety
///
/// Vectors **MUST** be equal length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_xany_avx512_fma_angular_hyperplane(x: &[f32], y: &[f32]) -> Vec<f32> {
    debug_assert_eq!(x.len(), y.len());

    let mut norm_x = f32_xany_avx512_fma_norm(x).sqrt();
    if norm_x.abs() < f32::EPSILON {
        norm_x = 1.0;
    }

    let mut norm_y = f32_xany_avx512_fma_norm(y).sqrt();
    if norm_y.abs() < f32::EPSILON {
        norm_y = 1.0;
    }

    let mut hyperplane = vec![0.0; x.len()];

    any_size_f32_hyperplane(x, y, norm_x, norm_y, hyperplane.as_mut_ptr());

    let mut norm_hyperplane = f32_xany_avx512_fma_norm(&hyperplane).sqrt();
    if norm_hyperplane.abs() < f32::EPSILON {
        norm_hyperplane = 1.0;
    }

    f32_xany_avx512_nofma_div_value(&mut hyperplane, norm_hyperplane);

    hyperplane
}

unsafe fn any_size_f32_hyperplane(
    x: &[f32],
    y: &[f32],
    norm_x: f32,
    norm_y: f32,
    hyperplane_ptr: *mut f32,
) {
    let len = x.len();
    let offset_from = len % 128;

    let x = x.as_ptr();
    let y = y.as_ptr();

    let inverse_norm_x = _mm512_set1_ps(1.0 / norm_x);
    let inverse_norm_y = _mm512_set1_ps(1.0 / norm_y);

    let mut i = 0;
    while i < (len - offset_from) {
        let results = execute_f32_x128_block_normal_vector(
            x.add(i),
            y.add(i),
            inverse_norm_x,
            inverse_norm_y,
        );
        ptr::copy_nonoverlapping(results.as_ptr(), hyperplane_ptr.add(i), results.len());

        i += 128;
    }

    while i < len {
        let n = len - i;

        let (x, y) = load_two_variable_size_avx512_ps(x.add(i), y.add(i), n);

        let normalized_x = _mm512_mul_ps(x, inverse_norm_x);
        let normalized_y = _mm512_mul_ps(y, inverse_norm_y);
        let diff = _mm512_sub_ps(normalized_x, normalized_y);

        copy_masked_avx512_ps_register_to(hyperplane_ptr.add(i), diff, n);

        i += 16;
    }
}

#[inline(always)]
unsafe fn execute_f32_x128_block_normal_vector(
    x: *const f32,
    y: *const f32,
    inverse_norm_x: __m512,
    inverse_norm_y: __m512,
) -> [f32; 128] {
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

    let normalized_x1 = _mm512_mul_ps(x1, inverse_norm_x);
    let normalized_x2 = _mm512_mul_ps(x2, inverse_norm_x);
    let normalized_x3 = _mm512_mul_ps(x3, inverse_norm_x);
    let normalized_x4 = _mm512_mul_ps(x4, inverse_norm_x);
    let normalized_x5 = _mm512_mul_ps(x5, inverse_norm_x);
    let normalized_x6 = _mm512_mul_ps(x6, inverse_norm_x);
    let normalized_x7 = _mm512_mul_ps(x7, inverse_norm_x);
    let normalized_x8 = _mm512_mul_ps(x8, inverse_norm_x);

    let normalized_y1 = _mm512_mul_ps(y1, inverse_norm_y);
    let normalized_y2 = _mm512_mul_ps(y2, inverse_norm_y);
    let normalized_y3 = _mm512_mul_ps(y3, inverse_norm_y);
    let normalized_y4 = _mm512_mul_ps(y4, inverse_norm_y);
    let normalized_y5 = _mm512_mul_ps(y5, inverse_norm_y);
    let normalized_y6 = _mm512_mul_ps(y6, inverse_norm_y);
    let normalized_y7 = _mm512_mul_ps(y7, inverse_norm_y);
    let normalized_y8 = _mm512_mul_ps(y8, inverse_norm_y);

    let diff1 = _mm512_sub_ps(normalized_x1, normalized_y1);
    let diff2 = _mm512_sub_ps(normalized_x2, normalized_y2);
    let diff3 = _mm512_sub_ps(normalized_x3, normalized_y3);
    let diff4 = _mm512_sub_ps(normalized_x4, normalized_y4);
    let diff5 = _mm512_sub_ps(normalized_x5, normalized_y5);
    let diff6 = _mm512_sub_ps(normalized_x6, normalized_y6);
    let diff7 = _mm512_sub_ps(normalized_x7, normalized_y7);
    let diff8 = _mm512_sub_ps(normalized_x8, normalized_y8);

    let lanes = [diff1, diff2, diff3, diff4, diff5, diff6, diff7, diff8];

    mem::transmute(lanes)
}

#[cfg(all(test, target_feature = "avx512f"))]
mod tests {
    use super::*;
    use crate::test_utils::{
        assert_is_close_vector,
        get_sample_vectors,
        simple_angular_hyperplane,
    };

    #[test]
    fn test_xconst_fma_angular_hyperplane() {
        let (x, y) = get_sample_vectors(1024);
        let hyperplane =
            unsafe { f32_xconst_avx512_fma_angular_hyperplane::<1024>(&x, &y) };
        let expected = simple_angular_hyperplane(&x, &y);
        assert_is_close_vector(&hyperplane, &expected);
    }

    #[test]
    fn test_xany_fma_angular_hyperplane() {
        let (x, y) = get_sample_vectors(517);
        let hyperplane = unsafe { f32_xany_avx512_fma_angular_hyperplane(&x, &y) };
        let expected = simple_angular_hyperplane(&x, &y);
        assert_is_close_vector(&hyperplane, &expected);
    }
}
