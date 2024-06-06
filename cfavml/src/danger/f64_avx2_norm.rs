use std::arch::x86_64::*;

use crate::danger::utils::{CHUNK_0, CHUNK_1};
use crate::danger::{offsets_avx2_pd, rollup_x8_pd, sum_avx2_pd};
use crate::math::*;

#[target_feature(enable = "avx2")]
#[inline]
/// Computes the squared norm of one `[f64; DIMS]` vector.
///
/// # Safety
///
/// DIMS **MUST** be a multiple of `32` and vector must be `DIMS` in length,
/// otherwise this routine will become immediately UB due to out of bounds pointer accesses.
///
/// This method assumes avx2 instructions are available, if this method is executed
/// on non-avx2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xconst_avx2_nofma_norm<const DIMS: usize>(x: &[f64]) -> f64 {
    let x = x.as_ptr();

    let mut acc1 = _mm256_setzero_pd();
    let mut acc2 = _mm256_setzero_pd();
    let mut acc3 = _mm256_setzero_pd();
    let mut acc4 = _mm256_setzero_pd();
    let mut acc5 = _mm256_setzero_pd();
    let mut acc6 = _mm256_setzero_pd();
    let mut acc7 = _mm256_setzero_pd();
    let mut acc8 = _mm256_setzero_pd();

    let mut i = 0;
    while i < DIMS {
        execute_f64_x32_nofma_block_norm(
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

        i += 32;
    }

    let acc = rollup_x8_pd(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8);
    sum_avx2_pd(acc)
}

#[target_feature(enable = "avx2")]
#[inline]
/// Computes the squared norm of one f64 vector.
///
/// This vector can be any size although it may perform worse than
/// the specialized size handling.
///
/// # Safety
///
/// This method assumes avx2 instructions are available, if this method is executed
/// on non-avx2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xany_avx2_nofma_norm(x: &[f64]) -> f64 {
    let len = x.len();
    let offset_from = len % 32;
    let mut total = 0.0;

    let x_ptr = x.as_ptr();
    let mut acc1 = _mm256_setzero_pd();
    let mut acc2 = _mm256_setzero_pd();
    let mut acc3 = _mm256_setzero_pd();
    let mut acc4 = _mm256_setzero_pd();
    let mut acc5 = _mm256_setzero_pd();
    let mut acc6 = _mm256_setzero_pd();
    let mut acc7 = _mm256_setzero_pd();
    let mut acc8 = _mm256_setzero_pd();

    let mut i = 0;
    while i < (len - offset_from) {
        execute_f64_x32_nofma_block_norm(
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

        i += 32;
    }

    if offset_from != 0 {
        let tail = offset_from % 4;

        while i < (len - tail) {
            let x = _mm256_loadu_pd(x_ptr.add(i));

            let res = _mm256_mul_pd(x, x);
            acc1 = _mm256_add_pd(acc1, res);

            i += 4;
        }

        while i < len {
            let x = *x.get_unchecked(i);
            total = AutoMath::add(total, AutoMath::mul(x, x));

            i += 1;
        }
    }

    let acc = rollup_x8_pd(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8);
    AutoMath::add(total, sum_avx2_pd(acc))
}

#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
/// Computes the squared norm of one `[f64; DIMS]` vector.
///
/// # Safety
///
/// DIMS **MUST** be a multiple of `32` and vector must be `DIMS` in length,
/// otherwise this routine will become immediately UB due to out of bounds pointer accesses.
///
/// This method assumes avx2 instructions are available, if this method is executed
/// on non-avx2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xconst_avx2_fma_norm<const DIMS: usize>(x: &[f64]) -> f64 {
    let x = x.as_ptr();

    let mut acc1 = _mm256_setzero_pd();
    let mut acc2 = _mm256_setzero_pd();
    let mut acc3 = _mm256_setzero_pd();
    let mut acc4 = _mm256_setzero_pd();
    let mut acc5 = _mm256_setzero_pd();
    let mut acc6 = _mm256_setzero_pd();
    let mut acc7 = _mm256_setzero_pd();
    let mut acc8 = _mm256_setzero_pd();

    let mut i = 0;
    while i < DIMS {
        execute_f64_x32_fma_block_norm(
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

        i += 32;
    }

    let acc = rollup_x8_pd(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8);
    sum_avx2_pd(acc)
}

#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
/// Computes the squared norm of one f64 vector.
///
/// This vector can be any size although it may perform worse than
/// the specialized size handling.
///
/// # Safety
///
/// This method assumes avx2 instructions are available, if this method is executed
/// on non-avx2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xany_avx2_fma_norm(x: &[f64]) -> f64 {
    let len = x.len();
    let offset_from = len % 32;
    let mut total = 0.0;

    let x_ptr = x.as_ptr();
    let mut acc1 = _mm256_setzero_pd();
    let mut acc2 = _mm256_setzero_pd();
    let mut acc3 = _mm256_setzero_pd();
    let mut acc4 = _mm256_setzero_pd();
    let mut acc5 = _mm256_setzero_pd();
    let mut acc6 = _mm256_setzero_pd();
    let mut acc7 = _mm256_setzero_pd();
    let mut acc8 = _mm256_setzero_pd();

    let mut i = 0;
    while i < (len - offset_from) {
        execute_f64_x32_fma_block_norm(
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

        i += 32;
    }

    if offset_from != 0 {
        let tail = offset_from % 4;

        while i < (len - tail) {
            let x = _mm256_loadu_pd(x_ptr.add(i));
            acc1 = _mm256_fmadd_pd(x, x, acc1);

            i += 4;
        }

        while i < len {
            let x = *x.get_unchecked(i);
            total = AutoMath::add(total, AutoMath::mul(x, x));

            i += 1;
        }
    }

    let acc = rollup_x8_pd(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8);
    AutoMath::add(total, sum_avx2_pd(acc))
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn execute_f64_x32_nofma_block_norm(
    x: *const f64,
    acc1: &mut __m256d,
    acc2: &mut __m256d,
    acc3: &mut __m256d,
    acc4: &mut __m256d,
    acc5: &mut __m256d,
    acc6: &mut __m256d,
    acc7: &mut __m256d,
    acc8: &mut __m256d,
) {
    let [x1, x2, x3, x4] = offsets_avx2_pd::<CHUNK_0>(x);
    let [x5, x6, x7, x8] = offsets_avx2_pd::<CHUNK_1>(x);

    let x1 = _mm256_loadu_pd(x1);
    let x2 = _mm256_loadu_pd(x2);
    let x3 = _mm256_loadu_pd(x3);
    let x4 = _mm256_loadu_pd(x4);
    let x5 = _mm256_loadu_pd(x5);
    let x6 = _mm256_loadu_pd(x6);
    let x7 = _mm256_loadu_pd(x7);
    let x8 = _mm256_loadu_pd(x8);

    let r1 = _mm256_mul_pd(x1, x1);
    let r2 = _mm256_mul_pd(x2, x2);
    let r3 = _mm256_mul_pd(x3, x3);
    let r4 = _mm256_mul_pd(x4, x4);
    let r5 = _mm256_mul_pd(x5, x5);
    let r6 = _mm256_mul_pd(x6, x6);
    let r7 = _mm256_mul_pd(x7, x7);
    let r8 = _mm256_mul_pd(x8, x8);

    *acc1 = _mm256_add_pd(*acc1, r1);
    *acc2 = _mm256_add_pd(*acc2, r2);
    *acc3 = _mm256_add_pd(*acc3, r3);
    *acc4 = _mm256_add_pd(*acc4, r4);
    *acc5 = _mm256_add_pd(*acc5, r5);
    *acc6 = _mm256_add_pd(*acc6, r6);
    *acc7 = _mm256_add_pd(*acc7, r7);
    *acc8 = _mm256_add_pd(*acc8, r8);
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn execute_f64_x32_fma_block_norm(
    x: *const f64,
    acc1: &mut __m256d,
    acc2: &mut __m256d,
    acc3: &mut __m256d,
    acc4: &mut __m256d,
    acc5: &mut __m256d,
    acc6: &mut __m256d,
    acc7: &mut __m256d,
    acc8: &mut __m256d,
) {
    let [x1, x2, x3, x4] = offsets_avx2_pd::<CHUNK_0>(x);
    let [x5, x6, x7, x8] = offsets_avx2_pd::<CHUNK_1>(x);

    let x1 = _mm256_loadu_pd(x1);
    let x2 = _mm256_loadu_pd(x2);
    let x3 = _mm256_loadu_pd(x3);
    let x4 = _mm256_loadu_pd(x4);
    let x5 = _mm256_loadu_pd(x5);
    let x6 = _mm256_loadu_pd(x6);
    let x7 = _mm256_loadu_pd(x7);
    let x8 = _mm256_loadu_pd(x8);

    *acc1 = _mm256_fmadd_pd(x1, x1, *acc1);
    *acc2 = _mm256_fmadd_pd(x2, x2, *acc2);
    *acc3 = _mm256_fmadd_pd(x3, x3, *acc3);
    *acc4 = _mm256_fmadd_pd(x4, x4, *acc4);
    *acc5 = _mm256_fmadd_pd(x5, x5, *acc5);
    *acc6 = _mm256_fmadd_pd(x6, x6, *acc6);
    *acc7 = _mm256_fmadd_pd(x7, x7, *acc7);
    *acc8 = _mm256_fmadd_pd(x8, x8, *acc8);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{assert_is_close, get_sample_vectors, simple_dot};

    #[test]
    fn test_xany_fma_norm() {
        let (x, _) = get_sample_vectors(127);
        let dist = unsafe { f64_xany_avx2_fma_norm(&x) };
        assert_is_close(dist as f32, simple_dot(&x, &x) as f32);
    }

    #[test]
    fn test_xany_nofma_norm() {
        let (x, _) = get_sample_vectors(127);
        let dist = unsafe { f64_xany_avx2_nofma_norm(&x) };
        assert_is_close(dist as f32, simple_dot(&x, &x) as f32);
    }

    #[test]
    fn test_xconst_fma_norm() {
        let (x, _) = get_sample_vectors(1024);
        let dist = unsafe { f64_xconst_avx2_fma_norm::<1024>(&x) };
        assert_is_close(dist as f32, simple_dot(&x, &x) as f32);
    }

    #[test]
    fn test_xconst_nofma_norm() {
        let (x, _) = get_sample_vectors(1024);
        let dist = unsafe { f64_xconst_avx2_nofma_norm::<1024>(&x) };
        assert_is_close(dist as f32, simple_dot(&x, &x) as f32);
    }
}
