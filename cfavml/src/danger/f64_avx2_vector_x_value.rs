use std::arch::x86_64::*;
use std::{mem, ptr};

use crate::danger::{copy_avx2_pd_register_to, offsets_avx2_pd, CHUNK_0, CHUNK_1};

macro_rules! complete_tail {
    ($offset_from:expr, $i:expr, $arr:ident, $value:expr, $value_reg:expr, $inst:ident, op = $op:tt) => {{
        if $offset_from != 0 {
            let len = $arr.len();
            let arr_ptr = $arr.as_mut_ptr();
            let remainder = $offset_from % 4;

            while $i < (len - remainder) {
                let x = _mm256_loadu_pd(arr_ptr.add($i));
                let r = $inst(x, $value_reg);
                copy_avx2_pd_register_to(arr_ptr.add($i), r);

                $i += 4;
            }

            while $i < len {
                let x = $arr.get_unchecked_mut($i);
                *x $op $value;
                $i += 1;
            }
        }
    }};
}

#[target_feature(enable = "avx2")]
#[inline]
/// Divides each element in the provided mutable `[f64; DIMS]` vector by `value`.
///
/// # Safety
///
/// Vectors **MUST** be `DIMS` elements in length and divisible by 32,
/// otherwise this function becomes immediately UB due to out of bounds
/// access.
///
/// This method assumes avx2 instructions are available, if this method is executed
/// on non-avx2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xconst_avx2_nofma_div_value<const DIMS: usize>(
    arr: &mut [f64],
    divider: f64,
) {
    f64_xconst_avx2_nofma_mul_value::<DIMS>(arr, 1.0 / divider)
}

#[target_feature(enable = "avx2")]
#[inline]
/// Divides each element in the provided mutable `f64` vector by `value`.
///
/// # Safety
///
/// This method assumes avx2 instructions are available, if this method is executed
/// on non-avx2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xany_avx2_nofma_div_value(arr: &mut [f64], divider: f64) {
    f64_xany_avx2_nofma_mul_value(arr, 1.0 / divider)
}

#[target_feature(enable = "avx2")]
#[inline]
/// Multiplies each element in the provided mutable `[f64; DIMS]` vector by `value`.
///
/// # Safety
///
/// Vectors **MUST** be `DIMS` elements in length and divisible by 32,
/// otherwise this function becomes immediately UB due to out of bounds
/// access.
///
/// This method assumes avx2 instructions are available, if this method is executed
/// on non-avx2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xconst_avx2_nofma_mul_value<const DIMS: usize>(
    arr: &mut [f64],
    multiplier: f64,
) {
    debug_assert_eq!(arr.len(), DIMS);
    debug_assert_eq!(DIMS % 32, 0, "Input dimensions must be multiple of 32");

    let multiplier = _mm256_set1_pd(multiplier);
    let arr = arr.as_mut_ptr();

    let mut i = 0;
    while i < DIMS {
        execute_f64_x64_mul(arr.add(i), multiplier);
        i += 32;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
/// Multiplies each element in the provided mutable `f64` vector by `value`.
///
/// # Safety
///
/// This method assumes avx2 instructions are available, if this method is executed
/// on non-avx2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xany_avx2_nofma_mul_value(arr: &mut [f64], multiplier: f64) {
    let len = arr.len();
    let offset_from = len % 32;

    let multiplier_reg = _mm256_set1_pd(multiplier);
    let arr_ptr = arr.as_mut_ptr();

    let mut i = 0;
    while i < (len - offset_from) {
        execute_f64_x64_mul(arr_ptr.add(i), multiplier_reg);
        i += 32;
    }

    complete_tail!(offset_from, i, arr, multiplier, multiplier_reg, _mm256_mul_pd, op = *=);
}

#[target_feature(enable = "avx2")]
#[inline]
/// Adds `value` to each element in the provided mutable `[f64; DIMS]` vector.
///
/// # Safety
///
/// Vectors **MUST** be `DIMS` elements in length and divisible by 32,
/// otherwise this function becomes immediately UB due to out of bounds
/// access.
///
/// This method assumes avx2 instructions are available, if this method is executed
/// on non-avx2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xconst_avx2_nofma_add_value<const DIMS: usize>(
    arr: &mut [f64],
    value: f64,
) {
    debug_assert_eq!(arr.len(), DIMS);
    debug_assert_eq!(DIMS % 32, 0, "Input dimensions must be multiple of 32");

    let value = _mm256_set1_pd(value);
    let arr = arr.as_mut_ptr();

    let mut i = 0;
    while i < DIMS {
        execute_f64_x64_add(arr.add(i), value);
        i += 32;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
/// Adds `value` to each element in the provided mutable `f64` vector.
///
/// # Safety
///
/// This method assumes avx2 instructions are available, if this method is executed
/// on non-avx2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xany_avx2_nofma_add_value(arr: &mut [f64], value: f64) {
    let len = arr.len();
    let offset_from = len % 32;

    let value_reg = _mm256_set1_pd(value);
    let arr_ptr = arr.as_mut_ptr();

    let mut i = 0;
    while i < (len - offset_from) {
        execute_f64_x64_add(arr_ptr.add(i), value_reg);
        i += 32;
    }

    complete_tail!(offset_from, i, arr, value, value_reg, _mm256_add_pd, op = +=);
}

#[target_feature(enable = "avx2")]
#[inline]
/// Subtracts `value` from each element in the provided mutable `[f64; DIMS]` vector.
///
/// # Safety
///
/// Vectors **MUST** be `DIMS` elements in length and divisible by 32,
/// otherwise this function becomes immediately UB due to out of bounds
/// access.
///
/// This method assumes avx2 instructions are available, if this method is executed
/// on non-avx2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xconst_avx2_nofma_sub_value<const DIMS: usize>(
    arr: &mut [f64],
    value: f64,
) {
    debug_assert_eq!(arr.len(), DIMS);
    debug_assert_eq!(DIMS % 32, 0, "Input dimensions must be multiple of 32");

    let arr = arr.as_mut_ptr();
    let value = _mm256_set1_pd(value);

    let mut i = 0;
    while i < DIMS {
        execute_f64_x64_sub(arr.add(i), value);
        i += 32;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
/// Subtracts `value` from each element in the provided mutable `f64` vector.
///
/// # Safety
///
/// This method assumes avx2 instructions are available, if this method is executed
/// on non-avx2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xany_avx2_nofma_sub_value(arr: &mut [f64], value: f64) {
    let len = arr.len();
    let offset_from = len % 32;

    let value_reg = _mm256_set1_pd(value);
    let arr_ptr = arr.as_mut_ptr();

    let mut i = 0;
    while i < (len - offset_from) {
        execute_f64_x64_sub(arr_ptr.add(i), value_reg);
        i += 32;
    }

    complete_tail!(offset_from, i, arr, value, value_reg, _mm256_sub_pd, op = -=);
}

#[inline(always)]
unsafe fn execute_f64_x64_mul(x: *mut f64, multiplier: __m256d) {
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

    let result1 = _mm256_mul_pd(x1, multiplier);
    let result2 = _mm256_mul_pd(x2, multiplier);
    let result3 = _mm256_mul_pd(x3, multiplier);
    let result4 = _mm256_mul_pd(x4, multiplier);
    let result5 = _mm256_mul_pd(x5, multiplier);
    let result6 = _mm256_mul_pd(x6, multiplier);
    let result7 = _mm256_mul_pd(x7, multiplier);
    let result8 = _mm256_mul_pd(x8, multiplier);

    let lanes = [
        result1, result2, result3, result4, result5, result6, result7, result8,
    ];

    let result = mem::transmute::<[__m256d; 8], [f64; 32]>(lanes);
    ptr::copy_nonoverlapping(result.as_ptr(), x, result.len());
}

#[inline(always)]
unsafe fn execute_f64_x64_add(x: *mut f64, value: __m256d) {
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

    let result1 = _mm256_add_pd(x1, value);
    let result2 = _mm256_add_pd(x2, value);
    let result3 = _mm256_add_pd(x3, value);
    let result4 = _mm256_add_pd(x4, value);
    let result5 = _mm256_add_pd(x5, value);
    let result6 = _mm256_add_pd(x6, value);
    let result7 = _mm256_add_pd(x7, value);
    let result8 = _mm256_add_pd(x8, value);

    let lanes = [
        result1, result2, result3, result4, result5, result6, result7, result8,
    ];

    let result = mem::transmute::<[__m256d; 8], [f64; 32]>(lanes);
    ptr::copy_nonoverlapping(result.as_ptr(), x, result.len());
}

#[inline(always)]
unsafe fn execute_f64_x64_sub(x: *mut f64, value: __m256d) {
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

    let result1 = _mm256_sub_pd(x1, value);
    let result2 = _mm256_sub_pd(x2, value);
    let result3 = _mm256_sub_pd(x3, value);
    let result4 = _mm256_sub_pd(x4, value);
    let result5 = _mm256_sub_pd(x5, value);
    let result6 = _mm256_sub_pd(x6, value);
    let result7 = _mm256_sub_pd(x7, value);
    let result8 = _mm256_sub_pd(x8, value);

    let lanes = [
        result1, result2, result3, result4, result5, result6, result7, result8,
    ];

    let result = mem::transmute::<[__m256d; 8], [f64; 32]>(lanes);
    ptr::copy_nonoverlapping(result.as_ptr(), x, result.len());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{assert_is_close_vector_f64, get_sample_vectors};

    #[test]
    fn test_f64_xany_div() {
        let value = 2.0;
        let (mut x, _) = get_sample_vectors(557);
        let expected = x.iter().copied().map(|v| v / value).collect::<Vec<_>>();
        unsafe { f64_xany_avx2_nofma_div_value(&mut x, value) };
        assert_is_close_vector_f64(&x, &expected);
    }

    #[test]
    fn test_f64_xany_mul() {
        let value = 2.0;
        let (mut x, _) = get_sample_vectors(557);
        let expected = x.iter().copied().map(|v| v * value).collect::<Vec<_>>();
        unsafe { f64_xany_avx2_nofma_mul_value(&mut x, value) };
        assert_is_close_vector_f64(&x, &expected);
    }

    #[test]
    fn test_f64_xany_add() {
        let value = 2.0;
        let (mut x, _) = get_sample_vectors(557);
        let expected = x.iter().copied().map(|v| v + value).collect::<Vec<_>>();
        unsafe { f64_xany_avx2_nofma_add_value(&mut x, value) };
        assert_is_close_vector_f64(&x, &expected);
    }

    #[test]
    fn test_f64_xany_sub() {
        let value = 2.0;
        let (mut x, _) = get_sample_vectors(557);
        let expected = x.iter().copied().map(|v| v - value).collect::<Vec<_>>();
        unsafe { f64_xany_avx2_nofma_sub_value(&mut x, value) };
        assert_is_close_vector_f64(&x, &expected);
    }

    #[test]
    fn test_f64_xconst_div() {
        let value = 2.0;
        let (mut x, _) = get_sample_vectors(512);
        let expected = x.iter().copied().map(|v| v / value).collect::<Vec<_>>();
        unsafe { f64_xconst_avx2_nofma_div_value::<512>(&mut x, value) };
        assert_is_close_vector_f64(&x, &expected);
    }

    #[test]
    fn test_f64_xconst_mul() {
        let value = 2.0;
        let (mut x, _) = get_sample_vectors(512);
        let expected = x.iter().copied().map(|v| v * value).collect::<Vec<_>>();
        unsafe { f64_xconst_avx2_nofma_mul_value::<512>(&mut x, value) };
        assert_is_close_vector_f64(&x, &expected);
    }

    #[test]
    fn test_f64_xconst_add() {
        let value = 2.0;
        let (mut x, _) = get_sample_vectors(512);
        let expected = x.iter().copied().map(|v| v + value).collect::<Vec<_>>();
        unsafe { f64_xconst_avx2_nofma_add_value::<512>(&mut x, value) };
        assert_is_close_vector_f64(&x, &expected);
    }

    #[test]
    fn test_f64_xconst_sub() {
        let value = 2.0;
        let (mut x, _) = get_sample_vectors(512);
        let expected = x.iter().copied().map(|v| v - value).collect::<Vec<_>>();
        unsafe { f64_xconst_avx2_nofma_sub_value::<512>(&mut x, value) };
        assert_is_close_vector_f64(&x, &expected);
    }
}
