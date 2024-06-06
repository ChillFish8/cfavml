use std::arch::x86_64::*;

use crate::danger::{offsets_avx512_ps, sum_avx512_x8_ps, CHUNK_0, CHUNK_1};

#[target_feature(enable = "avx512f")]
#[inline]
/// Computes the squared norm of one `[f32; DIMS]` vector.
///
/// # Safety
///
/// DIMS **MUST** be a multiple of `128` and vectors must be `DIMS` in length,
/// otherwise this routine will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_xconst_avx512_fma_norm<const DIMS: usize>(x: &[f32]) -> f32 {
    debug_assert_eq!(DIMS % 128, 0);
    debug_assert_eq!(x.len(), DIMS);

    let x = x.as_ptr();

    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut acc4 = _mm512_setzero_ps();
    let mut acc5 = _mm512_setzero_ps();
    let mut acc6 = _mm512_setzero_ps();
    let mut acc7 = _mm512_setzero_ps();
    let mut acc8 = _mm512_setzero_ps();

    let mut i = 0;
    while i < DIMS {
        execute_f32_x128_fma_block_norm(
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

        i += 128;
    }

    sum_avx512_x8_ps(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8)
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Computes the squared norm of one `f32` vector.
///
/// # Safety
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_xany_avx512_fma_norm(x: &[f32]) -> f32 {
    let len = x.len();
    let offset_from = len % 128;
    let x = x.as_ptr();

    let mut acc1 = _mm512_setzero_ps();
    let mut acc2 = _mm512_setzero_ps();
    let mut acc3 = _mm512_setzero_ps();
    let mut acc4 = _mm512_setzero_ps();
    let mut acc5 = _mm512_setzero_ps();
    let mut acc6 = _mm512_setzero_ps();
    let mut acc7 = _mm512_setzero_ps();
    let mut acc8 = _mm512_setzero_ps();

    let mut i = 0;
    while i < (len - offset_from) {
        execute_f32_x128_fma_block_norm(
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

        i += 128;
    }

    while i < len {
        let n = len - i;
        let addr = x.add(i);

        let x = if n < 16 {
            let mask = _bzhi_u32(0xFFFFFFFF, n as u32) as _;
            _mm512_maskz_loadu_ps(mask, addr)
        } else {
            _mm512_loadu_ps(addr)
        };

        acc1 = _mm512_fmadd_ps(x, x, acc1);

        i += 16
    }

    sum_avx512_x8_ps(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8)
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn execute_f32_x128_fma_block_norm(
    x: *const f32,
    acc1: &mut __m512,
    acc2: &mut __m512,
    acc3: &mut __m512,
    acc4: &mut __m512,
    acc5: &mut __m512,
    acc6: &mut __m512,
    acc7: &mut __m512,
    acc8: &mut __m512,
) {
    let [x1, x2, x3, x4] = offsets_avx512_ps::<CHUNK_0>(x);
    let [x5, x6, x7, x8] = offsets_avx512_ps::<CHUNK_1>(x);

    let x1 = _mm512_loadu_ps(x1);
    let x2 = _mm512_loadu_ps(x2);
    let x3 = _mm512_loadu_ps(x3);
    let x4 = _mm512_loadu_ps(x4);
    let x5 = _mm512_loadu_ps(x5);
    let x6 = _mm512_loadu_ps(x6);
    let x7 = _mm512_loadu_ps(x7);
    let x8 = _mm512_loadu_ps(x8);

    *acc1 = _mm512_fmadd_ps(x1, x1, *acc1);
    *acc2 = _mm512_fmadd_ps(x2, x2, *acc2);
    *acc3 = _mm512_fmadd_ps(x3, x3, *acc3);
    *acc4 = _mm512_fmadd_ps(x4, x4, *acc4);
    *acc5 = _mm512_fmadd_ps(x5, x5, *acc5);
    *acc6 = _mm512_fmadd_ps(x6, x6, *acc6);
    *acc7 = _mm512_fmadd_ps(x7, x7, *acc7);
    *acc8 = _mm512_fmadd_ps(x8, x8, *acc8);
}

#[cfg(all(test, target_feature = "avx512f"))]
mod tests {
    use super::*;
    use crate::test_utils::{assert_is_close, get_sample_vectors, simple_dot};

    #[test]
    fn test_xconst_fma_norm() {
        let (x, _) = get_sample_vectors(1024);
        let dist = unsafe { f32_xconst_avx512_fma_norm::<1024>(&x) };
        assert_is_close(dist, simple_dot(&x, &x));
    }

    #[test]
    fn test_xany_fma_norm() {
        let (x, _) = get_sample_vectors(547);
        let dist = unsafe { f32_xany_avx512_fma_norm(&x) };
        assert_is_close(dist, simple_dot(&x, &x));
    }
}
