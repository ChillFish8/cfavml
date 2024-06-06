use std::arch::x86_64::*;

use crate::danger::{
    load_two_variable_size_avx512_pd,
    offsets_avx512_pd,
    sum_avx512_x8_pd,
    CHUNK_0,
    CHUNK_1,
};

#[target_feature(enable = "avx512f")]
#[inline]
/// Computes the dot product of two `[f64; DIMS]` vectors.
///
/// # Safety
///
/// DIMS **MUST** be a multiple of `64` and vectors must be `DIMS` in length,
/// otherwise this routine will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f64_xconst_avx512_fma_dot<const DIMS: usize>(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(DIMS % 64, 0);
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), DIMS);

    let x = x.as_ptr();
    let y = y.as_ptr();

    let mut acc1 = _mm512_setzero_pd();
    let mut acc2 = _mm512_setzero_pd();
    let mut acc3 = _mm512_setzero_pd();
    let mut acc4 = _mm512_setzero_pd();
    let mut acc5 = _mm512_setzero_pd();
    let mut acc6 = _mm512_setzero_pd();
    let mut acc7 = _mm512_setzero_pd();
    let mut acc8 = _mm512_setzero_pd();

    let mut i = 0;
    while i < DIMS {
        execute_f64_x64_fma_block_dot_product(
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

    sum_avx512_x8_pd(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8)
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Computes the dot product of two `f64` vectors.
///
/// # Safety
///
/// Vectors **MUST** be equal length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f64_xany_avx512_fma_dot(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());

    let len = x.len();
    let offset_from = len % 64;

    let x = x.as_ptr();
    let y = y.as_ptr();

    let mut acc1 = _mm512_setzero_pd();
    let mut acc2 = _mm512_setzero_pd();
    let mut acc3 = _mm512_setzero_pd();
    let mut acc4 = _mm512_setzero_pd();
    let mut acc5 = _mm512_setzero_pd();
    let mut acc6 = _mm512_setzero_pd();
    let mut acc7 = _mm512_setzero_pd();
    let mut acc8 = _mm512_setzero_pd();

    let mut i = 0;
    while i < (len - offset_from) {
        execute_f64_x64_fma_block_dot_product(
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

    while i < len {
        let (x, y) = load_two_variable_size_avx512_pd(x.add(i), y.add(i), len - i);

        acc1 = _mm512_fmadd_pd(x, y, acc1);

        i += 8;
    }

    sum_avx512_x8_pd(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8)
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn execute_f64_x64_fma_block_dot_product(
    x: *const f64,
    y: *const f64,
    acc1: &mut __m512d,
    acc2: &mut __m512d,
    acc3: &mut __m512d,
    acc4: &mut __m512d,
    acc5: &mut __m512d,
    acc6: &mut __m512d,
    acc7: &mut __m512d,
    acc8: &mut __m512d,
) {
    let [x1, x2, x3, x4] = offsets_avx512_pd::<CHUNK_0>(x);
    let [x5, x6, x7, x8] = offsets_avx512_pd::<CHUNK_1>(x);

    let [y1, y2, y3, y4] = offsets_avx512_pd::<CHUNK_0>(y);
    let [y5, y6, y7, y8] = offsets_avx512_pd::<CHUNK_1>(y);

    let x1 = _mm512_loadu_pd(x1);
    let x2 = _mm512_loadu_pd(x2);
    let x3 = _mm512_loadu_pd(x3);
    let x4 = _mm512_loadu_pd(x4);
    let x5 = _mm512_loadu_pd(x5);
    let x6 = _mm512_loadu_pd(x6);
    let x7 = _mm512_loadu_pd(x7);
    let x8 = _mm512_loadu_pd(x8);

    let y1 = _mm512_loadu_pd(y1);
    let y2 = _mm512_loadu_pd(y2);
    let y3 = _mm512_loadu_pd(y3);
    let y4 = _mm512_loadu_pd(y4);
    let y5 = _mm512_loadu_pd(y5);
    let y6 = _mm512_loadu_pd(y6);
    let y7 = _mm512_loadu_pd(y7);
    let y8 = _mm512_loadu_pd(y8);

    *acc1 = _mm512_fmadd_pd(x1, y1, *acc1);
    *acc2 = _mm512_fmadd_pd(x2, y2, *acc2);
    *acc3 = _mm512_fmadd_pd(x3, y3, *acc3);
    *acc4 = _mm512_fmadd_pd(x4, y4, *acc4);
    *acc5 = _mm512_fmadd_pd(x5, y5, *acc5);
    *acc6 = _mm512_fmadd_pd(x6, y6, *acc6);
    *acc7 = _mm512_fmadd_pd(x7, y7, *acc7);
    *acc8 = _mm512_fmadd_pd(x8, y8, *acc8);
}

#[cfg(all(test, target_feature = "avx512f"))]
mod tests {
    use super::*;
    use crate::test_utils::{assert_is_close, get_sample_vectors, simple_dot};

    #[test]
    fn test_xconst_fma_dot() {
        let (x, y) = get_sample_vectors(1024);
        let dist = unsafe { f64_xconst_avx512_fma_dot::<1024>(&x, &y) };
        assert_is_close(dist as f32, simple_dot(&x, &y) as f32);
    }

    #[test]
    fn test_xany_fma_dot() {
        let (x, y) = get_sample_vectors(547);
        let dist = unsafe { f64_xany_avx512_fma_dot(&x, &y) };
        assert_is_close(dist as f32, simple_dot(&x, &y) as f32);
    }
}
