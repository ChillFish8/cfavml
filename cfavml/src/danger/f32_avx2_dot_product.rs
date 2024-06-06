use std::arch::x86_64::*;

use crate::danger::utils::{CHUNK_0, CHUNK_1};
use crate::danger::{offsets_avx2_ps, rollup_x8_ps, sum_avx2_ps};
use crate::math::*;

#[target_feature(enable = "avx2")]
#[inline]
/// Computes the dot product of two `[f32; DIMS]` vectors.
///
/// # Safety
///
/// DIMS **MUST** be a multiple of `64` and both vectors must be `DIMS` in length,
/// otherwise this routine will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_xconst_avx2_nofma_dot<const DIMS: usize>(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(DIMS % 64, 0);
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), DIMS);

    let x = x.as_ptr();
    let y = y.as_ptr();

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
        execute_f32_x64_nofma_block_dot_product(
            x.add(i),
            y.add(i),
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
/// Computes the dot product of two `f32` vectors.
///
/// # Safety
///
/// Vectors **MUST** be the same length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
pub unsafe fn f32_xany_avx2_nofma_dot(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());

    let len = x.len();
    let offset_from = len % 64;
    let mut total = 0.0;

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();

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
        execute_f32_x64_nofma_block_dot_product(
            x_ptr.add(i),
            y_ptr.add(i),
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
            let y = _mm256_loadu_ps(y_ptr.add(i));

            let res = _mm256_mul_ps(x, y);
            acc1 = _mm256_add_ps(acc1, res);

            i += 8;
        }

        for n in i..len {
            let x = *x.get_unchecked(n);
            let y = *y.get_unchecked(n);
            total = AutoMath::add(total, AutoMath::mul(x, y));
        }
    }

    let acc = rollup_x8_ps(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8);
    AutoMath::add(total, sum_avx2_ps(acc))
}

#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
/// Computes the dot product of two `[f32; DIMS]` vectors.
///
/// # Safety
///
/// DIMS **MUST** be a multiple of `64` and both vectors must be `DIMS` in length,
/// otherwise this routine will become immediately UB due to out of bounds pointer accesses.
pub unsafe fn f32_xconst_avx2_fma_dot<const DIMS: usize>(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(DIMS % 64, 0);
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), DIMS);

    let x = x.as_ptr();
    let y = y.as_ptr();

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
        execute_f32_x64_fma_block_dot_product(
            x.add(i),
            y.add(i),
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

#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
/// Computes the dot product of two `f32` vectors.
///
/// # Safety
///
/// Vectors **MUST** be the same length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
pub unsafe fn f32_xany_avx2_fma_dot(x: &[f32], y: &[f32]) -> f32 {
    use crate::math::*;

    debug_assert_eq!(x.len(), y.len());

    let len = x.len();
    let offset_from = len % 64;
    let mut total = 0.0;

    let x_ptr = x.as_ptr();
    let y_ptr = y.as_ptr();

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
        execute_f32_x64_fma_block_dot_product(
            x_ptr.add(i),
            y_ptr.add(i),
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
            let y = _mm256_loadu_ps(y_ptr.add(i));

            acc1 = _mm256_fmadd_ps(x, y, acc1);

            i += 8;
        }

        while i < len {
            let x = *x.get_unchecked(i);
            let y = *y.get_unchecked(i);
            total = AutoMath::add(total, AutoMath::mul(x, y));

            i += 1;
        }
    }

    let acc = rollup_x8_ps(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8);
    AutoMath::add(total, sum_avx2_ps(acc))
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn execute_f32_x64_nofma_block_dot_product(
    x: *const f32,
    y: *const f32,
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

    let r1 = _mm256_mul_ps(x1, y1);
    let r2 = _mm256_mul_ps(x2, y2);
    let r3 = _mm256_mul_ps(x3, y3);
    let r4 = _mm256_mul_ps(x4, y4);
    let r5 = _mm256_mul_ps(x5, y5);
    let r6 = _mm256_mul_ps(x6, y6);
    let r7 = _mm256_mul_ps(x7, y7);
    let r8 = _mm256_mul_ps(x8, y8);

    *acc1 = _mm256_add_ps(*acc1, r1);
    *acc2 = _mm256_add_ps(*acc2, r2);
    *acc3 = _mm256_add_ps(*acc3, r3);
    *acc4 = _mm256_add_ps(*acc4, r4);
    *acc5 = _mm256_add_ps(*acc5, r5);
    *acc6 = _mm256_add_ps(*acc6, r6);
    *acc7 = _mm256_add_ps(*acc7, r7);
    *acc8 = _mm256_add_ps(*acc8, r8);
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn execute_f32_x64_fma_block_dot_product(
    x: *const f32,
    y: *const f32,
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

    *acc1 = _mm256_fmadd_ps(x1, y1, *acc1);
    *acc2 = _mm256_fmadd_ps(x2, y2, *acc2);
    *acc3 = _mm256_fmadd_ps(x3, y3, *acc3);
    *acc4 = _mm256_fmadd_ps(x4, y4, *acc4);
    *acc5 = _mm256_fmadd_ps(x5, y5, *acc5);
    *acc6 = _mm256_fmadd_ps(x6, y6, *acc6);
    *acc7 = _mm256_fmadd_ps(x7, y7, *acc7);
    *acc8 = _mm256_fmadd_ps(x8, y8, *acc8);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{assert_is_close, get_sample_vectors, simple_dot};

    #[test]
    fn test_xany_fma_dot() {
        let (x, y) = get_sample_vectors(127);
        let dist = unsafe { f32_xany_avx2_fma_dot(&x, &y) };
        assert_is_close(dist, simple_dot(&x, &y))
    }

    #[test]
    fn test_xany_nofma_dot() {
        let (x, y) = get_sample_vectors(127);
        let dist = unsafe { f32_xany_avx2_nofma_dot(&x, &y) };
        assert_is_close(dist, simple_dot(&x, &y))
    }

    #[test]
    fn test_xconst_fma_dot() {
        let (x, y) = get_sample_vectors(1024);
        let dist = unsafe { f32_xconst_avx2_fma_dot::<1024>(&x, &y) };
        assert_is_close(dist, simple_dot(&x, &y))
    }

    #[test]
    fn test_xconst_nofma_dot() {
        let (x, y) = get_sample_vectors(1024);
        let dist = unsafe { f32_xconst_avx2_nofma_dot::<1024>(&x, &y) };
        assert_is_close(dist, simple_dot(&x, &y))
    }
}
