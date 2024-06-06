use std::arch::x86_64::*;
use std::{mem, ptr};

use crate::danger::{
    copy_masked_avx512_pd_register_to,
    load_one_variable_size_avx512_pd,
    offsets_avx512_pd,
    CHUNK_0,
    CHUNK_1,
};

#[target_feature(enable = "avx512f")]
#[inline]
/// Divides each element in the provided mutable `[f64; DIMS]` vector by `value`.
///
/// # Safety
///
/// Vectors **MUST** be `DIMS` elements in length and divisible by 64,
/// otherwise this function becomes immediately UB due to out of bounds
/// access.
///
/// This method assumes AVX512 instructions are available, if this method is executed
/// on non-AVX512 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xconst_avx512_nofma_div_value<const DIMS: usize>(
    arr: &mut [f64],
    divider: f64,
) {
    f64_xconst_avx512_nofma_mul_value::<DIMS>(arr, 1.0 / divider)
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Divides each element in the provided mutable `f64` vector by `value`.
///
/// # Safety
///
/// This method assumes AVX512 instructions are available, if this method is executed
/// on non-AVX512 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xany_avx512_nofma_div_value(arr: &mut [f64], divider: f64) {
    f64_xany_avx512_nofma_mul_value(arr, 1.0 / divider)
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Multiplies each element in the provided mutable `[f64; DIMS]` vector by `value`.
///
/// # Safety
///
/// Vectors **MUST** be `DIMS` elements in length and divisible by 64,
/// otherwise this function becomes immediately UB due to out of bounds
/// access.
///
/// This method assumes AVX512 instructions are available, if this method is executed
/// on non-AVX512 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xconst_avx512_nofma_mul_value<const DIMS: usize>(
    arr: &mut [f64],
    multiplier: f64,
) {
    debug_assert_eq!(arr.len(), DIMS);
    debug_assert_eq!(DIMS % 64, 0, "Input dimensions must be multiple of 64");

    let multiplier = _mm512_set1_pd(multiplier);
    let arr = arr.as_mut_ptr();

    let mut i = 0;
    while i < DIMS {
        execute_f64_x64_mul(arr.add(i), multiplier);
        i += 64;
    }
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Multiplies each element in the provided mutable `f64` vector by `value`.
///
/// # Safety
///
/// This method assumes AVX512 instructions are available, if this method is executed
/// on non-AVX512 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xany_avx512_nofma_mul_value(arr: &mut [f64], multiplier: f64) {
    let len = arr.len();
    let offset_from = len % 64;

    let multiplier = _mm512_set1_pd(multiplier);
    let arr = arr.as_mut_ptr();

    let mut i = 0;
    while i < (len - offset_from) {
        execute_f64_x64_mul(arr.add(i), multiplier);
        i += 64;
    }

    while i < len {
        let n = len - i;
        let arr = arr.add(i);
        let x = load_one_variable_size_avx512_pd(arr, n);
        let r = _mm512_mul_pd(x, multiplier);
        copy_masked_avx512_pd_register_to(arr, r, n);

        i += 8;
    }
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Adds `value` to each element in the provided mutable `[f64; DIMS]` vector.
///
/// # Safety
///
/// Vectors **MUST** be `DIMS` elements in length and divisible by 64,
/// otherwise this function becomes immediately UB due to out of bounds
/// access.
///
/// This method assumes AVX512 instructions are available, if this method is executed
/// on non-AVX512 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xconst_avx512_nofma_add_value<const DIMS: usize>(
    arr: &mut [f64],
    value: f64,
) {
    debug_assert_eq!(arr.len(), DIMS);
    debug_assert_eq!(DIMS % 64, 0, "Input dimensions must be multiple of 64");

    let value = _mm512_set1_pd(value);
    let arr = arr.as_mut_ptr();

    let mut i = 0;
    while i < DIMS {
        execute_f64_x64_add(arr.add(i), value);
        i += 64;
    }
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Adds `value` to each element in the provided mutable `f64` vector.
///
/// # Safety
///
/// This method assumes AVX512 instructions are available, if this method is executed
/// on non-AVX512 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xany_avx512_nofma_add_value(arr: &mut [f64], value: f64) {
    let len = arr.len();
    let offset_from = len % 64;

    let value = _mm512_set1_pd(value);
    let arr = arr.as_mut_ptr();

    let mut i = 0;
    while i < (len - offset_from) {
        execute_f64_x64_add(arr.add(i), value);
        i += 64;
    }

    while i < len {
        let n = len - i;
        let arr = arr.add(i);

        let r = if n < 8 {
            let mask = _bzhi_u32(0xFFFFFFFF, n as u32) as _;
            let x = _mm512_maskz_loadu_pd(mask, arr);
            _mm512_maskz_add_pd(mask, x, value)
        } else {
            let x = _mm512_loadu_pd(arr);
            _mm512_add_pd(x, value)
        };

        copy_masked_avx512_pd_register_to(arr, r, n);

        i += 8;
    }
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Subtracts `value` from each element in the provided mutable `[f64; DIMS]` vector.
///
/// # Safety
///
/// Vectors **MUST** be `DIMS` elements in length and divisible by 64,
/// otherwise this function becomes immediately UB due to out of bounds
/// access.
///
/// This method assumes AVX512 instructions are available, if this method is executed
/// on non-AVX512 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xconst_avx512_nofma_sub_value<const DIMS: usize>(
    arr: &mut [f64],
    value: f64,
) {
    debug_assert_eq!(arr.len(), DIMS);
    debug_assert_eq!(DIMS % 64, 0, "Input dimensions must be multiple of 64");

    let arr = arr.as_mut_ptr();
    let value = _mm512_set1_pd(value);

    let mut i = 0;
    while i < DIMS {
        execute_f64_x64_sub(arr.add(i), value);
        i += 64;
    }
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Subtracts `value` from each element in the provided mutable `f64` vector.
///
/// # Safety
///
/// This method assumes AVX512 instructions are available, if this method is executed
/// on non-AVX512 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xany_avx512_nofma_sub_value(arr: &mut [f64], value: f64) {
    let len = arr.len();
    let offset_from = len % 64;

    let value = _mm512_set1_pd(value);
    let arr = arr.as_mut_ptr();

    let mut i = 0;
    while i < (len - offset_from) {
        execute_f64_x64_sub(arr.add(i), value);
        i += 64;
    }

    while i < len {
        let n = len - i;
        let arr = arr.add(i);

        let r = if n < 8 {
            let mask = _bzhi_u32(0xFFFFFFFF, n as u32) as _;
            let x = _mm512_maskz_loadu_pd(mask, arr);
            _mm512_maskz_sub_pd(mask, x, value)
        } else {
            let x = _mm512_loadu_pd(arr);
            _mm512_sub_pd(x, value)
        };

        copy_masked_avx512_pd_register_to(arr, r, n);

        i += 8;
    }
}

#[inline(always)]
unsafe fn execute_f64_x64_mul(x: *mut f64, multiplier: __m512d) {
    let [x1, x2, x3, x4] = offsets_avx512_pd::<CHUNK_0>(x);
    let [x5, x6, x7, x8] = offsets_avx512_pd::<CHUNK_1>(x);

    let x1 = _mm512_loadu_pd(x1);
    let x2 = _mm512_loadu_pd(x2);
    let x3 = _mm512_loadu_pd(x3);
    let x4 = _mm512_loadu_pd(x4);
    let x5 = _mm512_loadu_pd(x5);
    let x6 = _mm512_loadu_pd(x6);
    let x7 = _mm512_loadu_pd(x7);
    let x8 = _mm512_loadu_pd(x8);

    let result1 = _mm512_mul_pd(x1, multiplier);
    let result2 = _mm512_mul_pd(x2, multiplier);
    let result3 = _mm512_mul_pd(x3, multiplier);
    let result4 = _mm512_mul_pd(x4, multiplier);
    let result5 = _mm512_mul_pd(x5, multiplier);
    let result6 = _mm512_mul_pd(x6, multiplier);
    let result7 = _mm512_mul_pd(x7, multiplier);
    let result8 = _mm512_mul_pd(x8, multiplier);

    let lanes = [
        result1, result2, result3, result4, result5, result6, result7, result8,
    ];

    let result = mem::transmute::<[__m512d; 8], [f64; 64]>(lanes);
    ptr::copy_nonoverlapping(result.as_ptr(), x, result.len());
}

#[inline(always)]
unsafe fn execute_f64_x64_add(x: *mut f64, value: __m512d) {
    let [x1, x2, x3, x4] = offsets_avx512_pd::<CHUNK_0>(x);
    let [x5, x6, x7, x8] = offsets_avx512_pd::<CHUNK_1>(x);

    let x1 = _mm512_loadu_pd(x1);
    let x2 = _mm512_loadu_pd(x2);
    let x3 = _mm512_loadu_pd(x3);
    let x4 = _mm512_loadu_pd(x4);
    let x5 = _mm512_loadu_pd(x5);
    let x6 = _mm512_loadu_pd(x6);
    let x7 = _mm512_loadu_pd(x7);
    let x8 = _mm512_loadu_pd(x8);

    let result1 = _mm512_add_pd(x1, value);
    let result2 = _mm512_add_pd(x2, value);
    let result3 = _mm512_add_pd(x3, value);
    let result4 = _mm512_add_pd(x4, value);
    let result5 = _mm512_add_pd(x5, value);
    let result6 = _mm512_add_pd(x6, value);
    let result7 = _mm512_add_pd(x7, value);
    let result8 = _mm512_add_pd(x8, value);

    let lanes = [
        result1, result2, result3, result4, result5, result6, result7, result8,
    ];

    let result = mem::transmute::<[__m512d; 8], [f64; 64]>(lanes);
    ptr::copy_nonoverlapping(result.as_ptr(), x, result.len());
}

#[inline(always)]
unsafe fn execute_f64_x64_sub(x: *mut f64, value: __m512d) {
    let [x1, x2, x3, x4] = offsets_avx512_pd::<CHUNK_0>(x);
    let [x5, x6, x7, x8] = offsets_avx512_pd::<CHUNK_1>(x);

    let x1 = _mm512_loadu_pd(x1);
    let x2 = _mm512_loadu_pd(x2);
    let x3 = _mm512_loadu_pd(x3);
    let x4 = _mm512_loadu_pd(x4);
    let x5 = _mm512_loadu_pd(x5);
    let x6 = _mm512_loadu_pd(x6);
    let x7 = _mm512_loadu_pd(x7);
    let x8 = _mm512_loadu_pd(x8);

    let result1 = _mm512_sub_pd(x1, value);
    let result2 = _mm512_sub_pd(x2, value);
    let result3 = _mm512_sub_pd(x3, value);
    let result4 = _mm512_sub_pd(x4, value);
    let result5 = _mm512_sub_pd(x5, value);
    let result6 = _mm512_sub_pd(x6, value);
    let result7 = _mm512_sub_pd(x7, value);
    let result8 = _mm512_sub_pd(x8, value);

    let lanes = [
        result1, result2, result3, result4, result5, result6, result7, result8,
    ];

    let result = mem::transmute::<[__m512d; 8], [f64; 64]>(lanes);
    ptr::copy_nonoverlapping(result.as_ptr(), x, result.len());
}

#[cfg(all(test, target_feature = "avx512f"))]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{assert_is_close_vector_f64, get_sample_vectors};

    #[test]
    fn test_f64_xany_div() {
        let value = 2.0;
        let (mut x, _) = get_sample_vectors(557);
        let expected = x.iter().copied().map(|v| v / value).collect::<Vec<_>>();
        unsafe { f64_xany_avx512_nofma_div_value(&mut x, value) };
        assert_is_close_vector_f64(&x, &expected);
    }

    #[test]
    fn test_f64_xany_mul() {
        let value = 2.0;
        let (mut x, _) = get_sample_vectors(557);
        let expected = x.iter().copied().map(|v| v * value).collect::<Vec<_>>();
        unsafe { f64_xany_avx512_nofma_mul_value(&mut x, value) };
        assert_is_close_vector_f64(&x, &expected);
    }

    #[test]
    fn test_f64_xany_add() {
        let value = 2.0;
        let (mut x, _) = get_sample_vectors(557);
        let expected = x.iter().copied().map(|v| v + value).collect::<Vec<_>>();
        unsafe { f64_xany_avx512_nofma_add_value(&mut x, value) };
        assert_is_close_vector_f64(&x, &expected);
    }

    #[test]
    fn test_f64_xany_sub() {
        let value = 2.0;
        let (mut x, _) = get_sample_vectors(557);
        let expected = x.iter().copied().map(|v| v - value).collect::<Vec<_>>();
        unsafe { f64_xany_avx512_nofma_sub_value(&mut x, value) };
        assert_is_close_vector_f64(&x, &expected);
    }

    #[test]
    fn test_f64_xconst_div() {
        let value = 2.0;
        let (mut x, _) = get_sample_vectors(512);
        let expected = x.iter().copied().map(|v| v / value).collect::<Vec<_>>();
        unsafe { f64_xconst_avx512_nofma_div_value::<512>(&mut x, value) };
        assert_is_close_vector_f64(&x, &expected);
    }

    #[test]
    fn test_f64_xconst_mul() {
        let value = 2.0;
        let (mut x, _) = get_sample_vectors(512);
        let expected = x.iter().copied().map(|v| v * value).collect::<Vec<_>>();
        unsafe { f64_xconst_avx512_nofma_mul_value::<512>(&mut x, value) };
        assert_is_close_vector_f64(&x, &expected);
    }

    #[test]
    fn test_f64_xconst_add() {
        let value = 2.0;
        let (mut x, _) = get_sample_vectors(512);
        let expected = x.iter().copied().map(|v| v + value).collect::<Vec<_>>();
        unsafe { f64_xconst_avx512_nofma_add_value::<512>(&mut x, value) };
        assert_is_close_vector_f64(&x, &expected);
    }

    #[test]
    fn test_f64_xconst_sub() {
        let value = 2.0;
        let (mut x, _) = get_sample_vectors(512);
        let expected = x.iter().copied().map(|v| v - value).collect::<Vec<_>>();
        unsafe { f64_xconst_avx512_nofma_sub_value::<512>(&mut x, value) };
        assert_is_close_vector_f64(&x, &expected);
    }
}
