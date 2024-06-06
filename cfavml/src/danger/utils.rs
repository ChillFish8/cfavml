use std::arch::x86_64::*;
use std::{mem, ptr};

use crate::math::Math;

pub const CHUNK_0: usize = 0;
pub const CHUNK_1: usize = 1;

#[allow(non_snake_case)]
pub const fn _MM_SHUFFLE(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[inline(always)]
pub fn cosine<T: Copy, M: Math<T>>(dot_product: T, norm_x: T, norm_y: T) -> T {
    if M::cmp_eq(norm_x, M::zero()) && M::cmp_eq(norm_y, M::zero()) {
        M::zero()
    } else if M::cmp_eq(norm_x, M::zero()) || M::cmp_eq(norm_y, M::zero()) {
        M::zero()
    } else {
        M::sub(
            M::one(),
            M::div(dot_product, M::sqrt(M::mul(norm_x, norm_y))),
        )
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
/// Performs a sum of all packed values in the provided [__m256] register
/// returning the resulting f32 value.
pub(crate) unsafe fn sum_avx2_ps(v: __m256) -> f32 {
    let left_half = _mm256_extractf128_ps::<1>(v);
    let right_half = _mm256_castps256_ps128(v);
    let sum_quad = _mm_add_ps(left_half, right_half);

    let left_half = sum_quad;
    let right_half = _mm_movehl_ps(sum_quad, sum_quad);
    let sum_dual = _mm_add_ps(left_half, right_half);

    let left_half = sum_dual;
    let right_half = _mm_shuffle_ps::<0x1>(sum_dual, sum_dual);
    let sum = _mm_add_ss(left_half, right_half);

    _mm_cvtss_f32(sum)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
/// Performs a sum of all packed values in the provided [__m256d] register
/// returning the resulting f32 value.
pub(crate) unsafe fn sum_avx2_pd(v: __m256d) -> f64 {
    let left_half = _mm256_extractf128_pd::<1>(v);
    let right_half = _mm256_castpd256_pd128(v);
    let sum_duo = _mm_add_pd(left_half, right_half);

    let undef = _mm_undefined_ps();
    let shuffle_tmp = _mm_movehl_ps(undef, _mm_castpd_ps(sum_duo));
    let shuffle = _mm_castps_pd(shuffle_tmp);
    _mm_cvtsd_f64(_mm_add_sd(sum_duo, shuffle))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
/// Rolls up 8 [__m256] registers into 1 summing them together.
pub(crate) unsafe fn rollup_x8_ps(
    mut acc1: __m256,
    acc2: __m256,
    mut acc3: __m256,
    acc4: __m256,
    mut acc5: __m256,
    acc6: __m256,
    mut acc7: __m256,
    acc8: __m256,
) -> __m256 {
    acc1 = _mm256_add_ps(acc1, acc2);
    acc3 = _mm256_add_ps(acc3, acc4);
    acc5 = _mm256_add_ps(acc5, acc6);
    acc7 = _mm256_add_ps(acc7, acc8);

    acc1 = _mm256_add_ps(acc1, acc3);
    acc5 = _mm256_add_ps(acc5, acc7);

    _mm256_add_ps(acc1, acc5)
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
/// Rolls up 8 [__m256] registers into 1 summing them together.
pub(crate) unsafe fn sum_avx512_x8_ps(
    mut acc1: __m512,
    acc2: __m512,
    mut acc3: __m512,
    acc4: __m512,
    mut acc5: __m512,
    acc6: __m512,
    mut acc7: __m512,
    acc8: __m512,
) -> f32 {
    acc1 = _mm512_add_ps(acc1, acc2);
    acc3 = _mm512_add_ps(acc3, acc4);
    acc5 = _mm512_add_ps(acc5, acc6);
    acc7 = _mm512_add_ps(acc7, acc8);

    acc1 = _mm512_add_ps(acc1, acc3);
    acc5 = _mm512_add_ps(acc5, acc7);

    acc1 = _mm512_add_ps(acc1, acc5);
    _mm512_reduce_add_ps(acc1)
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
/// Rolls up 8 [__m256d] registers into 1 summing them together.
pub(crate) unsafe fn sum_avx512_x8_pd(
    mut acc1: __m512d,
    acc2: __m512d,
    mut acc3: __m512d,
    acc4: __m512d,
    mut acc5: __m512d,
    acc6: __m512d,
    mut acc7: __m512d,
    acc8: __m512d,
) -> f64 {
    acc1 = _mm512_add_pd(acc1, acc2);
    acc3 = _mm512_add_pd(acc3, acc4);
    acc5 = _mm512_add_pd(acc5, acc6);
    acc7 = _mm512_add_pd(acc7, acc8);

    acc1 = _mm512_add_pd(acc1, acc3);
    acc5 = _mm512_add_pd(acc5, acc7);

    acc1 = _mm512_add_pd(acc1, acc5);
    _mm512_reduce_add_pd(acc1)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
/// Rolls up 3 [__m256d] registers into 1 summing them together.
pub(crate) unsafe fn rollup_x8_pd(
    mut acc1: __m256d,
    acc2: __m256d,
    mut acc3: __m256d,
    acc4: __m256d,
    mut acc5: __m256d,
    acc6: __m256d,
    mut acc7: __m256d,
    acc8: __m256d,
) -> __m256d {
    acc1 = _mm256_add_pd(acc1, acc2);
    acc3 = _mm256_add_pd(acc3, acc4);
    acc5 = _mm256_add_pd(acc5, acc6);
    acc7 = _mm256_add_pd(acc7, acc8);

    acc1 = _mm256_add_pd(acc1, acc3);
    acc5 = _mm256_add_pd(acc5, acc7);

    _mm256_add_pd(acc1, acc5)
}

#[inline(always)]
pub(crate) unsafe fn offsets_avx2_ps<const CHUNK: usize>(
    ptr: *const f32,
) -> [*const f32; 4] {
    [
        ptr.add(CHUNK * 32),
        ptr.add((CHUNK * 32) + 8),
        ptr.add((CHUNK * 32) + 16),
        ptr.add((CHUNK * 32) + 24),
    ]
}

#[inline(always)]
pub(crate) unsafe fn offsets_avx2_pd<const CHUNK: usize>(
    ptr: *const f64,
) -> [*const f64; 4] {
    [
        ptr.add(CHUNK * 16),
        ptr.add((CHUNK * 16) + 4),
        ptr.add((CHUNK * 16) + 8),
        ptr.add((CHUNK * 16) + 12),
    ]
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
#[inline(always)]
pub(crate) unsafe fn offsets_avx512_ps<const CHUNK: usize>(
    ptr: *const f32,
) -> [*const f32; 4] {
    [
        ptr.add(CHUNK * 64),
        ptr.add((CHUNK * 64) + 16),
        ptr.add((CHUNK * 64) + 32),
        ptr.add((CHUNK * 64) + 48),
    ]
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
#[inline(always)]
pub(crate) unsafe fn offsets_avx512_pd<const CHUNK: usize>(
    ptr: *const f64,
) -> [*const f64; 4] {
    [
        ptr.add(CHUNK * 32),
        ptr.add((CHUNK * 32) + 8),
        ptr.add((CHUNK * 32) + 16),
        ptr.add((CHUNK * 32) + 24),
    ]
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
/// Sums 8 scalar accumulators into one `T` value.
pub fn rollup_scalar_x8<T: Copy, M: Math<T>>(
    mut acc1: T,
    acc2: T,
    mut acc3: T,
    acc4: T,
    mut acc5: T,
    acc6: T,
    mut acc7: T,
    acc8: T,
) -> T {
    acc1 = M::add(acc1, acc2);
    acc3 = M::add(acc3, acc4);
    acc5 = M::add(acc5, acc6);
    acc7 = M::add(acc7, acc8);

    acc1 = M::add(acc1, acc3);
    acc5 = M::add(acc5, acc7);

    M::add(acc1, acc5)
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub(crate) unsafe fn load_two_variable_size_avx512_ps(
    x: *const f32,
    y: *const f32,
    n: usize,
) -> (__m512, __m512) {
    if n < 16 {
        let mask = _bzhi_u32(0xFFFFFFFF, n as u32) as _;
        let x = _mm512_maskz_loadu_ps(mask, x);
        let y = _mm512_maskz_loadu_ps(mask, y);
        (x, y)
    } else {
        let x = _mm512_loadu_ps(x);
        let y = _mm512_loadu_ps(y);
        (x, y)
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub(crate) unsafe fn load_two_variable_size_avx512_pd(
    x: *const f64,
    y: *const f64,
    n: usize,
) -> (__m512d, __m512d) {
    if n < 8 {
        let mask = _bzhi_u32(0xFFFFFFFF, n as u32) as _;
        let x = _mm512_maskz_loadu_pd(mask, x);
        let y = _mm512_maskz_loadu_pd(mask, y);
        (x, y)
    } else {
        let x = _mm512_loadu_pd(x);
        let y = _mm512_loadu_pd(y);
        (x, y)
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub(crate) unsafe fn load_one_variable_size_avx512_ps(
    x: *const f32,
    n: usize,
) -> __m512 {
    if n < 16 {
        let mask = _bzhi_u32(0xFFFFFFFF, n as u32) as _;
        _mm512_maskz_loadu_ps(mask, x)
    } else {
        _mm512_loadu_ps(x)
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub(crate) unsafe fn load_one_variable_size_avx512_pd(
    x: *const f64,
    n: usize,
) -> __m512d {
    if n < 16 {
        let mask = _bzhi_u32(0xFFFFFFFF, n as u32) as _;
        _mm512_maskz_loadu_pd(mask, x)
    } else {
        _mm512_loadu_pd(x)
    }
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
#[inline(always)]
/// Copies the data from the given `reg` into `arr` for upto `len` elements.
///
/// NOTE:
/// This will implicitly cap the number of elements to `min(len, 16)` to prevent
/// going out of bounds on the register.
pub(crate) unsafe fn copy_masked_avx512_ps_register_to(
    arr: *mut f32,
    reg: __m512,
    len: usize,
) {
    let result = mem::transmute::<__m512, [f32; 16]>(reg);
    ptr::copy_nonoverlapping(result.as_ptr(), arr, std::cmp::min(16, len));
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
#[inline(always)]
/// Copies the data from the given `reg` into `arr` for upto `len` elements.
///
/// NOTE:
/// This will implicitly cap the number of elements to `min(len, 8)` to prevent
/// going out of bounds on the register.
pub(crate) unsafe fn copy_masked_avx512_pd_register_to(
    arr: *mut f64,
    reg: __m512d,
    len: usize,
) {
    let result = mem::transmute::<__m512d, [f64; 8]>(reg);
    ptr::copy_nonoverlapping(result.as_ptr(), arr, std::cmp::min(8, len));
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
/// Copies the data from the given `reg` into `arr`.
pub(crate) unsafe fn copy_avx2_ps_register_to(arr: *mut f32, reg: __m256) {
    let result = mem::transmute::<__m256, [f32; 8]>(reg);
    ptr::copy_nonoverlapping(result.as_ptr(), arr, result.len());
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
/// Copies the data from the given `reg` into `arr`.
pub(crate) unsafe fn copy_avx2_pd_register_to(arr: *mut f64, reg: __m256d) {
    let result = mem::transmute::<__m256d, [f64; 4]>(reg);
    ptr::copy_nonoverlapping(result.as_ptr(), arr, result.len());
}

#[cfg(test)]
mod tests {
    use std::array;

    use super::*;
    use crate::math::AutoMath;

    #[test]
    fn test_avx2_offsets() {
        let x: [f32; 32] = array::from_fn(|i| i as f32);
        let [p1, p2, p3, p4] = unsafe { offsets_avx2_ps::<CHUNK_0>(x.as_ptr()) };
        assert_eq!(x[0..].as_ptr(), p1);
        assert_eq!(x[8..].as_ptr(), p2);
        assert_eq!(x[16..].as_ptr(), p3);
        assert_eq!(x[24..].as_ptr(), p4);
    }

    #[test]
    fn test_avx512_offsets() {
        let x: [f32; 64] = array::from_fn(|i| i as f32);
        let [p1, p2, p3, p4] = unsafe { offsets_avx512_ps::<CHUNK_0>(x.as_ptr()) };
        assert_eq!(x[0..].as_ptr(), p1);
        assert_eq!(x[16..].as_ptr(), p2);
        assert_eq!(x[32..].as_ptr(), p3);
        assert_eq!(x[48..].as_ptr(), p4);
    }

    #[test]
    fn test_rollup_scalar_x8() {
        let res =
            rollup_scalar_x8::<f32, AutoMath>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(res, 0.0);

        let res =
            rollup_scalar_x8::<f32, AutoMath>(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        assert_eq!(res, 8.0);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn test_sum_avx2() {
        unsafe {
            let acc = _mm256_setzero_ps();
            let res = sum_avx2_ps(acc);
            assert_eq!(res, 0.0);

            let acc = _mm256_set1_ps(1.0);
            let res = sum_avx2_ps(acc);
            assert_eq!(res, 8.0);

            let acc = _mm256_setzero_pd();
            let res = sum_avx2_pd(acc);
            assert_eq!(res, 0.0);

            let acc = _mm256_set1_pd(1.0);
            let res = sum_avx2_pd(acc);
            assert_eq!(res, 4.0);
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn test_rollup_avx2_x8() {
        unsafe {
            let acc1 = _mm256_setzero_ps();
            let acc2 = _mm256_setzero_ps();
            let acc3 = _mm256_setzero_ps();
            let acc4 = _mm256_setzero_ps();
            let acc5 = _mm256_setzero_ps();
            let acc6 = _mm256_setzero_ps();
            let acc7 = _mm256_setzero_ps();
            let acc8 = _mm256_setzero_ps();
            let res = rollup_x8_ps(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8);
            let res = sum_avx2_ps(res);
            assert_eq!(res, 0.0);

            let acc1 = _mm256_set1_ps(1.0);
            let acc2 = _mm256_set1_ps(1.0);
            let acc3 = _mm256_set1_ps(1.0);
            let acc4 = _mm256_set1_ps(1.0);
            let acc5 = _mm256_set1_ps(1.0);
            let acc6 = _mm256_set1_ps(1.0);
            let acc7 = _mm256_set1_ps(1.0);
            let acc8 = _mm256_set1_ps(1.0);
            let res = rollup_x8_ps(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8);
            let res = sum_avx2_ps(res);
            assert_eq!(res, 64.0);
        }
    }
}
