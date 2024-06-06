use std::arch::x86_64::*;
use std::{mem, ptr};

use crate::danger::{copy_avx2_ps_register_to, offsets_avx2_ps, CHUNK_0, CHUNK_1};

macro_rules! complete_tail {
    ($offset_from:expr, $i:expr, $arr:ident, $value:expr, $value_reg:expr, $inst:ident, op = $op:tt) => {{
        if $offset_from != 0 {
            let len = $arr.len();
            let arr_ptr = $arr.as_mut_ptr();
            let remainder = $offset_from % 8;

            while $i < (len - remainder) {
                let x = _mm256_loadu_ps(arr_ptr.add($i));
                let r = $inst(x, $value_reg);
                copy_avx2_ps_register_to(arr_ptr.add($i), r);

                $i += 8;
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
/// Divides each element in the provided mutable `[f32; DIMS]` vector by `value`.
///
/// # Safety
///
/// Vectors **MUST** be `DIMS` elements in length and divisible by 64,
/// otherwise this function becomes immediately UB due to out of bounds
/// access.
///
/// This method assumes avx2 instructions are available, if this method is executed
/// on non-avx2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xconst_avx2_nofma_div_value<const DIMS: usize>(
    arr: &mut [f32],
    divider: f32,
) {
    f32_xconst_avx2_nofma_mul_value::<DIMS>(arr, 1.0 / divider)
}

#[target_feature(enable = "avx2")]
#[inline]
/// Divides each element in the provided mutable `f32` vector by `value`.
///
/// # Safety
///
/// This method assumes avx2 instructions are available, if this method is executed
/// on non-avx2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xany_avx2_nofma_div_value(arr: &mut [f32], divider: f32) {
    f32_xany_avx2_nofma_mul_value(arr, 1.0 / divider)
}

#[target_feature(enable = "avx2")]
#[inline]
/// Multiplies each element in the provided mutable `[f32; DIMS]` vector by `value`.
///
/// # Safety
///
/// Vectors **MUST** be `DIMS` elements in length and divisible by 64,
/// otherwise this function becomes immediately UB due to out of bounds
/// access.
///
/// This method assumes avx2 instructions are available, if this method is executed
/// on non-avx2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xconst_avx2_nofma_mul_value<const DIMS: usize>(
    arr: &mut [f32],
    multiplier: f32,
) {
    debug_assert_eq!(arr.len(), DIMS);
    debug_assert_eq!(DIMS % 64, 0, "Input dimensions must be multiple of 64");

    let multiplier = _mm256_set1_ps(multiplier);
    let arr = arr.as_mut_ptr();

    let mut i = 0;
    while i < DIMS {
        execute_f32_x64_mul(arr.add(i), multiplier);
        i += 64;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
/// Multiplies each element in the provided mutable `f32` vector by `value`.
///
/// # Safety
///
/// This method assumes avx2 instructions are available, if this method is executed
/// on non-avx2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xany_avx2_nofma_mul_value(arr: &mut [f32], multiplier: f32) {
    let len = arr.len();
    let offset_from = len % 64;

    let multiplier_reg = _mm256_set1_ps(multiplier);
    let arr_ptr = arr.as_mut_ptr();

    let mut i = 0;
    while i < (len - offset_from) {
        execute_f32_x64_mul(arr_ptr.add(i), multiplier_reg);
        i += 64;
    }

    complete_tail!(offset_from, i, arr, multiplier, multiplier_reg, _mm256_mul_ps, op = *=);
}

#[target_feature(enable = "avx2")]
#[inline]
/// Adds `value` to each element in the provided mutable `[f32; DIMS]` vector.
///
/// # Safety
///
/// Vectors **MUST** be `DIMS` elements in length and divisible by 64,
/// otherwise this function becomes immediately UB due to out of bounds
/// access.
///
/// This method assumes avx2 instructions are available, if this method is executed
/// on non-avx2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xconst_avx2_nofma_add_value<const DIMS: usize>(
    arr: &mut [f32],
    value: f32,
) {
    debug_assert_eq!(arr.len(), DIMS);
    debug_assert_eq!(DIMS % 64, 0, "Input dimensions must be multiple of 64");

    let value = _mm256_set1_ps(value);
    let arr = arr.as_mut_ptr();

    let mut i = 0;
    while i < DIMS {
        execute_f32_x64_add(arr.add(i), value);
        i += 64;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
/// Adds `value` to each element in the provided mutable `f32` vector.
///
/// # Safety
///
/// This method assumes avx2 instructions are available, if this method is executed
/// on non-avx2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xany_avx2_nofma_add_value(arr: &mut [f32], value: f32) {
    let len = arr.len();
    let offset_from = len % 64;

    let value_reg = _mm256_set1_ps(value);
    let arr_ptr = arr.as_mut_ptr();

    let mut i = 0;
    while i < (len - offset_from) {
        execute_f32_x64_add(arr_ptr.add(i), value_reg);
        i += 64;
    }

    complete_tail!(offset_from, i, arr, value, value_reg, _mm256_add_ps, op = +=);
}

#[target_feature(enable = "avx2")]
#[inline]
/// Subtracts `value` from each element in the provided mutable `[f32; DIMS]` vector.
///
/// # Safety
///
/// Vectors **MUST** be `DIMS` elements in length and divisible by 64,
/// otherwise this function becomes immediately UB due to out of bounds
/// access.
///
/// This method assumes avx2 instructions are available, if this method is executed
/// on non-avx2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xconst_avx2_nofma_sub_value<const DIMS: usize>(
    arr: &mut [f32],
    value: f32,
) {
    debug_assert_eq!(arr.len(), DIMS);
    debug_assert_eq!(DIMS % 64, 0, "Input dimensions must be multiple of 64");

    let arr = arr.as_mut_ptr();
    let value = _mm256_set1_ps(value);

    let mut i = 0;
    while i < DIMS {
        execute_f32_x64_sub(arr.add(i), value);
        i += 64;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
/// Subtracts `value` from each element in the provided mutable `f32` vector.
///
/// # Safety
///
/// This method assumes avx2 instructions are available, if this method is executed
/// on non-avx2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xany_avx2_nofma_sub_value(arr: &mut [f32], value: f32) {
    let len = arr.len();
    let offset_from = len % 64;

    let value_reg = _mm256_set1_ps(value);
    let arr_ptr = arr.as_mut_ptr();

    let mut i = 0;
    while i < (len - offset_from) {
        execute_f32_x64_sub(arr_ptr.add(i), value_reg);
        i += 64;
    }

    complete_tail!(offset_from, i, arr, value, value_reg, _mm256_sub_ps, op = -=);
}

#[inline(always)]
unsafe fn execute_f32_x64_mul(x: *mut f32, multiplier: __m256) {
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

    let result1 = _mm256_mul_ps(x1, multiplier);
    let result2 = _mm256_mul_ps(x2, multiplier);
    let result3 = _mm256_mul_ps(x3, multiplier);
    let result4 = _mm256_mul_ps(x4, multiplier);
    let result5 = _mm256_mul_ps(x5, multiplier);
    let result6 = _mm256_mul_ps(x6, multiplier);
    let result7 = _mm256_mul_ps(x7, multiplier);
    let result8 = _mm256_mul_ps(x8, multiplier);

    let lanes = [
        result1, result2, result3, result4, result5, result6, result7, result8,
    ];

    let result = mem::transmute::<[__m256; 8], [f32; 64]>(lanes);
    ptr::copy_nonoverlapping(result.as_ptr(), x, result.len());
}

#[inline(always)]
unsafe fn execute_f32_x64_add(x: *mut f32, value: __m256) {
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

    let result1 = _mm256_add_ps(x1, value);
    let result2 = _mm256_add_ps(x2, value);
    let result3 = _mm256_add_ps(x3, value);
    let result4 = _mm256_add_ps(x4, value);
    let result5 = _mm256_add_ps(x5, value);
    let result6 = _mm256_add_ps(x6, value);
    let result7 = _mm256_add_ps(x7, value);
    let result8 = _mm256_add_ps(x8, value);

    let lanes = [
        result1, result2, result3, result4, result5, result6, result7, result8,
    ];

    let result = mem::transmute::<[__m256; 8], [f32; 64]>(lanes);
    ptr::copy_nonoverlapping(result.as_ptr(), x, result.len());
}

#[inline(always)]
unsafe fn execute_f32_x64_sub(x: *mut f32, value: __m256) {
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

    let result1 = _mm256_sub_ps(x1, value);
    let result2 = _mm256_sub_ps(x2, value);
    let result3 = _mm256_sub_ps(x3, value);
    let result4 = _mm256_sub_ps(x4, value);
    let result5 = _mm256_sub_ps(x5, value);
    let result6 = _mm256_sub_ps(x6, value);
    let result7 = _mm256_sub_ps(x7, value);
    let result8 = _mm256_sub_ps(x8, value);

    let lanes = [
        result1, result2, result3, result4, result5, result6, result7, result8,
    ];

    let result = mem::transmute::<[__m256; 8], [f32; 64]>(lanes);
    ptr::copy_nonoverlapping(result.as_ptr(), x, result.len());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{assert_is_close_vector, get_sample_vectors};

    #[test]
    fn test_f32_xany_div() {
        let value = 2.0;
        let (mut x, _) = get_sample_vectors(557);
        let expected = x.iter().copied().map(|v| v / value).collect::<Vec<_>>();
        unsafe { f32_xany_avx2_nofma_div_value(&mut x, value) };
        assert_is_close_vector(&x, &expected);
    }

    #[test]
    fn test_f32_xany_mul() {
        let value = 2.0;
        let (mut x, _) = get_sample_vectors(557);
        let expected = x.iter().copied().map(|v| v * value).collect::<Vec<_>>();
        unsafe { f32_xany_avx2_nofma_mul_value(&mut x, value) };
        assert_is_close_vector(&x, &expected);
    }

    #[test]
    fn test_f32_xany_add() {
        let value = 2.0;
        let (mut x, _) = get_sample_vectors(557);
        let expected = x.iter().copied().map(|v| v + value).collect::<Vec<_>>();
        unsafe { f32_xany_avx2_nofma_add_value(&mut x, value) };
        assert_is_close_vector(&x, &expected);
    }

    #[test]
    fn test_f32_xany_sub() {
        let value = 2.0;
        let (mut x, _) = get_sample_vectors(557);
        let expected = x.iter().copied().map(|v| v - value).collect::<Vec<_>>();
        unsafe { f32_xany_avx2_nofma_sub_value(&mut x, value) };
        assert_is_close_vector(&x, &expected);
    }

    #[test]
    fn test_f32_xconst_div() {
        let value = 2.0;
        let (mut x, _) = get_sample_vectors(512);
        let expected = x.iter().copied().map(|v| v / value).collect::<Vec<_>>();
        unsafe { f32_xconst_avx2_nofma_div_value::<512>(&mut x, value) };
        assert_is_close_vector(&x, &expected);
    }

    #[test]
    fn test_f32_xconst_mul() {
        let value = 2.0;
        let (mut x, _) = get_sample_vectors(512);
        let expected = x.iter().copied().map(|v| v * value).collect::<Vec<_>>();
        unsafe { f32_xconst_avx2_nofma_mul_value::<512>(&mut x, value) };
        assert_is_close_vector(&x, &expected);
    }

    #[test]
    fn test_f32_xconst_add() {
        let value = 2.0;
        let (mut x, _) = get_sample_vectors(512);
        let expected = x.iter().copied().map(|v| v + value).collect::<Vec<_>>();
        unsafe { f32_xconst_avx2_nofma_add_value::<512>(&mut x, value) };
        assert_is_close_vector(&x, &expected);
    }

    #[test]
    fn test_f32_xconst_sub() {
        let value = 2.0;
        let (mut x, _) = get_sample_vectors(512);
        let expected = x.iter().copied().map(|v| v - value).collect::<Vec<_>>();
        unsafe { f32_xconst_avx2_nofma_sub_value::<512>(&mut x, value) };
        assert_is_close_vector(&x, &expected);
    }
}
