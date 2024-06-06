use std::arch::x86_64::*;
use std::{mem, ptr};

use crate::danger::{copy_avx2_pd_register_to, offsets_avx2_pd, CHUNK_0, CHUNK_1};

macro_rules! x64_op_inplace {
    ($x:expr, $y:expr, $op:ident) => {{
        let [x1, x2, x3, x4] = offsets_avx2_pd::<CHUNK_0>($x);
        let [x5, x6, x7, x8] = offsets_avx2_pd::<CHUNK_1>($x);

        let [y1, y2, y3, y4] = offsets_avx2_pd::<CHUNK_0>($y);
        let [y5, y6, y7, y8] = offsets_avx2_pd::<CHUNK_1>($y);

        let x1 = _mm256_loadu_pd(x1);
        let x2 = _mm256_loadu_pd(x2);
        let x3 = _mm256_loadu_pd(x3);
        let x4 = _mm256_loadu_pd(x4);
        let x5 = _mm256_loadu_pd(x5);
        let x6 = _mm256_loadu_pd(x6);
        let x7 = _mm256_loadu_pd(x7);
        let x8 = _mm256_loadu_pd(x8);

        let y1 = _mm256_loadu_pd(y1);
        let y2 = _mm256_loadu_pd(y2);
        let y3 = _mm256_loadu_pd(y3);
        let y4 = _mm256_loadu_pd(y4);
        let y5 = _mm256_loadu_pd(y5);
        let y6 = _mm256_loadu_pd(y6);
        let y7 = _mm256_loadu_pd(y7);
        let y8 = _mm256_loadu_pd(y8);

        let r1 = $op(x1, y1);
        let r2 = $op(x2, y2);
        let r3 = $op(x3, y3);
        let r4 = $op(x4, y4);
        let r5 = $op(x5, y5);
        let r6 = $op(x6, y6);
        let r7 = $op(x7, y7);
        let r8 = $op(x8, y8);

        write_x64_block($x, r1, r2, r3, r4, r5, r6, r7, r8);
    }};
}

macro_rules! execute_tail_op_inplace {
    ($len:expr, $i:expr, $x:ident, $y:ident, $op:ident, op = $raw_op:tt) => {{
        let x_ptr = $x.as_mut_ptr();
        let y_ptr = $y.as_ptr();

        let remainder = $len % 4;
        while $i < ($len - remainder) {
            let x = _mm256_loadu_pd(x_ptr.add($i));
            let y = _mm256_loadu_pd(y_ptr.add($i));

            let reg = $op(x, y);
            copy_avx2_pd_register_to(x_ptr.add($i), reg);

            $i += 4;
        }

        for i in $i..$len {
            let v = *$x.get_unchecked(i);
            *$x.get_unchecked_mut(i) = v $raw_op *$y.get_unchecked(i);
        }

    }};
}

#[target_feature(enable = "avx2")]
#[inline]
/// Divides each the input mutable vector `x` by the elements provided by `y`
/// element wise.
///
/// ```py
/// D: int
/// x: [f64; D]
/// y: [f64; D]
///
/// for i in 0..D:
///     x[i] = x[i] / y[i]
/// ```
///
/// # Safety
///
/// `DIMS` **MUST** be a multiple of `32`, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// This method assumes AVX2 instructions are available, if this method is executed
/// on non-AVX2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xconst_avx2_nofma_div_vertical<const DIMS: usize>(
    x: &mut [f64],
    y: &[f64],
) {
    debug_assert_eq!(DIMS % 32, 0, "DIMS must be multiple of 32");
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), DIMS);

    let x = x.as_mut_ptr();
    let y = y.as_ptr();

    let mut i = 0;
    while i < DIMS {
        x64_op_inplace!(x.add(i), y.add(i), _mm256_div_pd);

        i += 32;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
/// Divides each the input mutable vector `x` by the elements provided by `y`
/// element wise.
///
/// ```py
/// D: int
/// x: [f64; D]
/// y: [f64; D]
///
/// for i in 0..D:
///     x[i] = x[i] / y[i]
/// ```
///
/// # Safety
///
/// This method assumes AVX2 instructions are available, if this method is executed
/// on non-AVX2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xany_avx2_nofma_div_vertical(x: &mut [f64], y: &[f64]) {
    debug_assert_eq!(x.len(), y.len());
    let len = x.len();
    let offset_from = len % 32;

    let x_ptr = x.as_mut_ptr();
    let y_ptr = y.as_ptr();

    let mut i = 0;
    while i < (len - offset_from) {
        x64_op_inplace!(x_ptr.add(i), y_ptr.add(i), _mm256_div_pd);

        i += 32;
    }

    if offset_from != 0 {
        execute_tail_op_inplace!(len, i, x, y, _mm256_div_pd, op = /);
    }
}

#[target_feature(enable = "avx2")]
#[inline]
/// Multiplies each the input mutable vector `x` by the elements provided by `y`
/// element wise.
///
/// ```py
/// D: int
/// x: [f64; D]
/// y: [f64; D]
///
/// for i in 0..D:
///     x[i] = x[i] * y[i]
/// ```
///
/// # Safety
///
/// `DIMS` **MUST** be a multiple of `32`, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// This method assumes AVX2 instructions are available, if this method is executed
/// on non-AVX2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xconst_avx2_nofma_mul_vertical<const DIMS: usize>(
    x: &mut [f64],
    y: &[f64],
) {
    debug_assert_eq!(DIMS % 32, 0, "DIMS must be multiple of 32");
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), DIMS);

    let x = x.as_mut_ptr();
    let y = y.as_ptr();

    let mut i = 0;
    while i < DIMS {
        x64_op_inplace!(x.add(i), y.add(i), _mm256_mul_pd);

        i += 32;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
/// Multiplies each the input mutable vector `x` by the elements provided by `y`
/// element wise.
///
/// ```py
/// D: int
/// x: [f64; D]
/// y: [f64; D]
///
/// for i in 0..D:
///     x[i] = x[i] * y[i]
/// ```
///
/// # Safety
///
/// This method assumes AVX2 instructions are available, if this method is executed
/// on non-AVX2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xany_avx2_nofma_mul_vertical(x: &mut [f64], y: &[f64]) {
    debug_assert_eq!(x.len(), y.len());
    let len = x.len();
    let offset_from = len % 32;

    let x_ptr = x.as_mut_ptr();
    let y_ptr = y.as_ptr();

    let mut i = 0;
    while i < (len - offset_from) {
        x64_op_inplace!(x_ptr.add(i), y_ptr.add(i), _mm256_mul_pd);

        i += 32;
    }

    if offset_from != 0 {
        execute_tail_op_inplace!(len, i, x, y, _mm256_mul_pd, op = *);
    }
}

#[target_feature(enable = "avx2")]
#[inline]
/// Adds each the input mutable vector `x` with the elements provided by `y`
/// element wise.
///
/// ```py
/// D: int
/// x: [f64; D]
/// y: [f64; D]
///
/// for i in 0..D:
///     x[i] = x[i] + y[i]
/// ```
///
/// # Safety
///
/// `DIMS` **MUST** be a multiple of `32`, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// This method assumes AVX2 instructions are available, if this method is executed
/// on non-AVX2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xconst_avx2_nofma_add_vertical<const DIMS: usize>(
    x: &mut [f64],
    y: &[f64],
) {
    debug_assert_eq!(DIMS % 32, 0, "DIMS must be multiple of 32");
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), DIMS);

    let x = x.as_mut_ptr();
    let y = y.as_ptr();

    let mut i = 0;
    while i < DIMS {
        x64_op_inplace!(x.add(i), y.add(i), _mm256_add_pd);

        i += 32;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
/// Adds each the input mutable vector `x` with the elements provided by `y`
/// element wise.
///
/// ```py
/// D: int
/// x: [f64; D]
/// y: [f64; D]
///
/// for i in 0..D:
///     x[i] = x[i] + y[i]
/// ```
///
/// # Safety
///
/// This method assumes AVX2 instructions are available, if this method is executed
/// on non-AVX2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xany_avx2_nofma_add_vertical(x: &mut [f64], y: &[f64]) {
    debug_assert_eq!(x.len(), y.len());
    let len = x.len();
    let offset_from = len % 32;

    let x_ptr = x.as_mut_ptr();
    let y_ptr = y.as_ptr();

    let mut i = 0;
    while i < (len - offset_from) {
        x64_op_inplace!(x_ptr.add(i), y_ptr.add(i), _mm256_add_pd);

        i += 32;
    }

    if offset_from != 0 {
        execute_tail_op_inplace!(len, i, x, y, _mm256_add_pd, op = +);
    }
}

#[target_feature(enable = "avx2")]
#[inline]
/// Subtracts each the input mutable vector `x` by the elements provided by `y`
/// element wise.
///
/// ```py
/// D: int
/// x: [f64; D]
/// y: [f64; D]
///
/// for i in 0..D:
///     x[i] = x[i] - y[i]
/// ```
///
/// # Safety
///
/// `DIMS` **MUST** be a multiple of `32`, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// This method assumes AVX2 instructions are available, if this method is executed
/// on non-AVX2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xconst_avx2_nofma_sub_vertical<const DIMS: usize>(
    x: &mut [f64],
    y: &[f64],
) {
    debug_assert_eq!(DIMS % 32, 0, "DIMS must be multiple of 32");
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), DIMS);

    let x = x.as_mut_ptr();
    let y = y.as_ptr();

    let mut i = 0;
    while i < DIMS {
        x64_op_inplace!(x.add(i), y.add(i), _mm256_sub_pd);

        i += 32;
    }
}

#[target_feature(enable = "avx2")]
#[inline]
/// Subtracts each the input mutable vector `x` by the elements provided by `y`
/// element wise.
///
/// ```py
/// D: int
/// x: [f64; D]
/// y: [f64; D]
///
/// for i in 0..D:
///     x[i] = x[i] - y[i]
/// ```
///
/// # Safety
///
/// This method assumes AVX2 instructions are available, if this method is executed
/// on non-AVX2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f64_xany_avx2_nofma_sub_vertical(x: &mut [f64], y: &[f64]) {
    debug_assert_eq!(x.len(), y.len());
    let len = x.len();
    let offset_from = len % 32;

    let x_ptr = x.as_mut_ptr();
    let y_ptr = y.as_ptr();

    let mut i = 0;
    while i < (len - offset_from) {
        x64_op_inplace!(x_ptr.add(i), y_ptr.add(i), _mm256_sub_pd);

        i += 32;
    }

    if offset_from != 0 {
        execute_tail_op_inplace!(len, i, x, y, _mm256_sub_pd, op = -);
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn write_x64_block(
    x: *mut f64,
    r1: __m256d,
    r2: __m256d,
    r3: __m256d,
    r4: __m256d,
    r5: __m256d,
    r6: __m256d,
    r7: __m256d,
    r8: __m256d,
) {
    let merged = [r1, r2, r3, r4, r5, r6, r7, r8];

    let results = mem::transmute::<[__m256d; 8], [f64; 32]>(merged);
    ptr::copy_nonoverlapping(results.as_ptr(), x, results.len());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{assert_is_close_vector_f64, get_sample_vectors};

    #[test]
    fn test_xconst_div_vertical() {
        let (mut x, y) = get_sample_vectors(512);

        let expected_res = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| x / y)
            .collect::<Vec<f64>>();

        unsafe { f64_xconst_avx2_nofma_div_vertical::<512>(&mut x, &y) };

        assert_is_close_vector_f64(&x, &expected_res);
    }

    #[test]
    fn test_xconst_mul_vertical() {
        let (mut x, y) = get_sample_vectors(512);

        let expected_res = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| x * y)
            .collect::<Vec<f64>>();

        unsafe { f64_xconst_avx2_nofma_mul_vertical::<512>(&mut x, &y) };

        assert_is_close_vector_f64(&x, &expected_res);
    }

    #[test]
    fn test_xconst_add_vertical() {
        let (mut x, y) = get_sample_vectors(512);

        let expected_res = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| x + y)
            .collect::<Vec<f64>>();

        unsafe { f64_xconst_avx2_nofma_add_vertical::<512>(&mut x, &y) };

        assert_is_close_vector_f64(&x, &expected_res);
    }

    #[test]
    fn test_xconst_sub_vertical() {
        let (mut x, y) = get_sample_vectors(512);

        let expected_res = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| x - y)
            .collect::<Vec<f64>>();

        unsafe { f64_xconst_avx2_nofma_sub_vertical::<512>(&mut x, &y) };

        assert_is_close_vector_f64(&x, &expected_res);
    }

    #[test]
    fn test_xany_div_vertical() {
        let (mut x, y) = get_sample_vectors(543);

        let expected_res = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| x / y)
            .collect::<Vec<f64>>();

        unsafe { f64_xany_avx2_nofma_div_vertical(&mut x, &y) };

        assert_is_close_vector_f64(&x, &expected_res);
    }

    #[test]
    fn test_xany_mul_vertical() {
        let (mut x, y) = get_sample_vectors(543);

        let expected_res = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| x * y)
            .collect::<Vec<f64>>();

        unsafe { f64_xany_avx2_nofma_mul_vertical(&mut x, &y) };

        assert_is_close_vector_f64(&x, &expected_res);
    }

    #[test]
    fn test_xany_add_vertical() {
        let (mut x, y) = get_sample_vectors(543);

        let expected_res = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| x + y)
            .collect::<Vec<f64>>();

        unsafe { f64_xany_avx2_nofma_add_vertical(&mut x, &y) };

        assert_is_close_vector_f64(&x, &expected_res);
    }

    #[test]
    fn test_xany_sub_vertical() {
        let (mut x, y) = get_sample_vectors(543);

        let expected_res = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| x - y)
            .collect::<Vec<f64>>();

        unsafe { f64_xany_avx2_nofma_sub_vertical(&mut x, &y) };

        assert_is_close_vector_f64(&x, &expected_res);
    }
}
