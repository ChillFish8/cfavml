use std::arch::x86_64::*;
use std::{mem, ptr};

use crate::danger::{
    copy_masked_avx512_ps_register_to,
    load_two_variable_size_avx512_ps,
    offsets_avx512_ps,
    CHUNK_0,
    CHUNK_1,
};

macro_rules! x128_op_inplace {
    ($x:expr, $y:expr, $op:ident) => {{
        let [x1, x2, x3, x4] = offsets_avx512_ps::<CHUNK_0>($x);
        let [x5, x6, x7, x8] = offsets_avx512_ps::<CHUNK_1>($x);

        let [y1, y2, y3, y4] = offsets_avx512_ps::<CHUNK_0>($y);
        let [y5, y6, y7, y8] = offsets_avx512_ps::<CHUNK_1>($y);

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

        let r1 = $op(x1, y1);
        let r2 = $op(x2, y2);
        let r3 = $op(x3, y3);
        let r4 = $op(x4, y4);
        let r5 = $op(x5, y5);
        let r6 = $op(x6, y6);
        let r7 = $op(x7, y7);
        let r8 = $op(x8, y8);

        write_x128_block($x, r1, r2, r3, r4, r5, r6, r7, r8);
    }};
}

macro_rules! execute_tail_op_inplace {
    ($len:expr, $i:expr, $x_ptr:ident, $y_ptr:ident, $op:ident) => {{
        while $i < $len {
            let n = $len - $i;

            let (x, y) =
                load_two_variable_size_avx512_ps($x_ptr.add($i), $y_ptr.add($i), n);

            let reg = $op(x, y);

            copy_masked_avx512_ps_register_to($x_ptr.add($i), reg, n);

            $i += 16;
        }
    }};
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Divides each the input mutable vector `x` by the elements provided by `y`
/// element wise.
///
/// ```py
/// D: int
/// x: [f32; D]
/// y: [f32; D]
///
/// for i in 0..D:
///     x[i] = x[i] / y[i]
/// ```
///
/// # Safety
///
/// `DIMS` **MUST** be a multiple of `128`, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// This method assumes AVX2 instructions are available, if this method is executed
/// on non-AVX2 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xconst_avx512_nofma_div_vertical<const DIMS: usize>(
    x: &mut [f32],
    y: &[f32],
) {
    debug_assert_eq!(DIMS % 128, 0, "DIMS must be multiple of 128");
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), DIMS);

    let x = x.as_mut_ptr();
    let y = y.as_ptr();

    let mut i = 0;
    while i < DIMS {
        x128_op_inplace!(x.add(i), y.add(i), _mm512_div_ps);

        i += 128;
    }
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Divides each the input mutable vector `x` by the elements provided by `y`
/// element wise.
///
/// ```py
/// D: int
/// x: [f32; D]
/// y: [f32; D]
///
/// for i in 0..D:
///     x[i] = x[i] / y[i]
/// ```
///
/// # Safety
///
/// This method assumes AVX512 instructions are available, if this method is executed
/// on non-AVX512 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xany_avx512_nofma_div_vertical(x: &mut [f32], y: &[f32]) {
    debug_assert_eq!(x.len(), y.len());
    let len = x.len();
    let offset_from = len % 128;

    let x = x.as_mut_ptr();
    let y = y.as_ptr();

    let mut i = 0;
    while i < (len - offset_from) {
        x128_op_inplace!(x.add(i), y.add(i), _mm512_div_ps);

        i += 128;
    }

    execute_tail_op_inplace!(len, i, x, y, _mm512_div_ps);
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Multiplies each the input mutable vector `x` by the elements provided by `y`
/// element wise.
///
/// ```py
/// D: int
/// x: [f32; D]
/// y: [f32; D]
///
/// for i in 0..D:
///     x[i] = x[i] * y[i]
/// ```
///
/// # Safety
///
/// `DIMS` **MUST** be a multiple of `128`, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// This method assumes AVX512 instructions are available, if this method is executed
/// on non-AVX512 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xconst_avx512_nofma_mul_vertical<const DIMS: usize>(
    x: &mut [f32],
    y: &[f32],
) {
    debug_assert_eq!(DIMS % 128, 0, "DIMS must be multiple of 128");
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), DIMS);

    let x = x.as_mut_ptr();
    let y = y.as_ptr();

    let mut i = 0;
    while i < DIMS {
        x128_op_inplace!(x.add(i), y.add(i), _mm512_mul_ps);

        i += 128;
    }
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Multiplies each the input mutable vector `x` by the elements provided by `y`
/// element wise.
///
/// ```py
/// D: int
/// x: [f32; D]
/// y: [f32; D]
///
/// for i in 0..D:
///     x[i] = x[i] * y[i]
/// ```
///
/// # Safety
///
/// This method assumes AVX512 instructions are available, if this method is executed
/// on non-AVX512 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xany_avx512_nofma_mul_vertical(x: &mut [f32], y: &[f32]) {
    debug_assert_eq!(x.len(), y.len());
    let len = x.len();
    let offset_from = len % 128;

    let x = x.as_mut_ptr();
    let y = y.as_ptr();

    let mut i = 0;
    while i < (len - offset_from) {
        x128_op_inplace!(x.add(i), y.add(i), _mm512_mul_ps);

        i += 128;
    }

    execute_tail_op_inplace!(len, i, x, y, _mm512_mul_ps);
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Adds each the input mutable vector `x` with the elements provided by `y`
/// element wise.
///
/// ```py
/// D: int
/// x: [f32; D]
/// y: [f32; D]
///
/// for i in 0..D:
///     x[i] = x[i] + y[i]
/// ```
///
/// # Safety
///
/// `DIMS` **MUST** be a multiple of `128`, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// This method assumes AVX512 instructions are available, if this method is executed
/// on non-AVX512 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xconst_avx512_nofma_add_vertical<const DIMS: usize>(
    x: &mut [f32],
    y: &[f32],
) {
    debug_assert_eq!(DIMS % 128, 0, "DIMS must be multiple of 128");
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), DIMS);

    let x = x.as_mut_ptr();
    let y = y.as_ptr();

    let mut i = 0;
    while i < DIMS {
        x128_op_inplace!(x.add(i), y.add(i), _mm512_add_ps);

        i += 128;
    }
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Adds each the input mutable vector `x` with the elements provided by `y`
/// element wise.
///
/// ```py
/// D: int
/// x: [f32; D]
/// y: [f32; D]
///
/// for i in 0..D:
///     x[i] = x[i] + y[i]
/// ```
///
/// # Safety
///
/// This method assumes AVX512 instructions are available, if this method is executed
/// on non-AVX512 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xany_avx512_nofma_add_vertical(x: &mut [f32], y: &[f32]) {
    debug_assert_eq!(x.len(), y.len());
    let len = x.len();
    let offset_from = len % 128;

    let x = x.as_mut_ptr();
    let y = y.as_ptr();

    let mut i = 0;
    while i < (len - offset_from) {
        x128_op_inplace!(x.add(i), y.add(i), _mm512_add_ps);

        i += 128;
    }

    execute_tail_op_inplace!(len, i, x, y, _mm512_add_ps);
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Subtracts each the input mutable vector `x` by the elements provided by `y`
/// element wise.
///
/// ```py
/// D: int
/// x: [f32; D]
/// y: [f32; D]
///
/// for i in 0..D:
///     x[i] = x[i] - y[i]
/// ```
///
/// # Safety
///
/// `DIMS` **MUST** be a multiple of `128`, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// This method assumes AVX512 instructions are available, if this method is executed
/// on non-AVX512 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xconst_avx512_nofma_sub_vertical<const DIMS: usize>(
    x: &mut [f32],
    y: &[f32],
) {
    debug_assert_eq!(DIMS % 128, 0, "DIMS must be multiple of 128");
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), DIMS);

    let x = x.as_mut_ptr();
    let y = y.as_ptr();

    let mut i = 0;
    while i < DIMS {
        x128_op_inplace!(x.add(i), y.add(i), _mm512_sub_ps);

        i += 128;
    }
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Subtracts each the input mutable vector `x` by the elements provided by `y`
/// element wise.
///
/// ```py
/// D: int
/// x: [f32; D]
/// y: [f32; D]
///
/// for i in 0..D:
///     x[i] = x[i] - y[i]
/// ```
///
/// # Safety
///
/// This method assumes AVX512 instructions are available, if this method is executed
/// on non-AVX512 enabled systems, it will lead to an `ILLEGAL_INSTRUCTION` error.
pub unsafe fn f32_xany_avx512_nofma_sub_vertical(x: &mut [f32], y: &[f32]) {
    debug_assert_eq!(x.len(), y.len());
    let len = x.len();
    let offset_from = len % 128;

    let x = x.as_mut_ptr();
    let y = y.as_ptr();

    let mut i = 0;
    while i < (len - offset_from) {
        x128_op_inplace!(x.add(i), y.add(i), _mm512_sub_ps);

        i += 128;
    }

    execute_tail_op_inplace!(len, i, x, y, _mm512_sub_ps);
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn write_x128_block(
    x: *mut f32,
    r1: __m512,
    r2: __m512,
    r3: __m512,
    r4: __m512,
    r5: __m512,
    r6: __m512,
    r7: __m512,
    r8: __m512,
) {
    let merged = [r1, r2, r3, r4, r5, r6, r7, r8];

    let results = mem::transmute::<[__m512; 8], [f32; 128]>(merged);
    ptr::copy_nonoverlapping(results.as_ptr(), x, results.len());
}

#[cfg(all(test, target_feature = "avx512f"))]
mod tests {
    use super::*;
    use crate::test_utils::{assert_is_close_vector, get_sample_vectors};

    #[test]
    fn test_xconst_div_vertical() {
        let (mut x, y) = get_sample_vectors(512);

        let expected_res = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| x / y)
            .collect::<Vec<f32>>();

        unsafe { f32_xconst_avx512_nofma_div_vertical::<512>(&mut x, &y) };

        assert_is_close_vector(&x, &expected_res);
    }

    #[test]
    fn test_xconst_mul_vertical() {
        let (mut x, y) = get_sample_vectors(512);

        let expected_res = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| x * y)
            .collect::<Vec<f32>>();

        unsafe { f32_xconst_avx512_nofma_mul_vertical::<512>(&mut x, &y) };

        assert_is_close_vector(&x, &expected_res);
    }

    #[test]
    fn test_xconst_add_vertical() {
        let (mut x, y) = get_sample_vectors(512);

        let expected_res = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| x + y)
            .collect::<Vec<f32>>();

        unsafe { f32_xconst_avx512_nofma_add_vertical::<512>(&mut x, &y) };

        assert_is_close_vector(&x, &expected_res);
    }

    #[test]
    fn test_xconst_sub_vertical() {
        let (mut x, y) = get_sample_vectors(512);

        let expected_res = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| x - y)
            .collect::<Vec<f32>>();

        unsafe { f32_xconst_avx512_nofma_sub_vertical::<512>(&mut x, &y) };

        assert_is_close_vector(&x, &expected_res);
    }

    #[test]
    fn test_xany_div_vertical() {
        let (mut x, y) = get_sample_vectors(543);

        let expected_res = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| x / y)
            .collect::<Vec<f32>>();

        unsafe { f32_xany_avx512_nofma_div_vertical(&mut x, &y) };

        assert_is_close_vector(&x, &expected_res);
    }

    #[test]
    fn test_xany_mul_vertical() {
        let (mut x, y) = get_sample_vectors(543);

        let expected_res = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| x * y)
            .collect::<Vec<f32>>();

        unsafe { f32_xany_avx512_nofma_mul_vertical(&mut x, &y) };

        assert_is_close_vector(&x, &expected_res);
    }

    #[test]
    fn test_xany_add_vertical() {
        let (mut x, y) = get_sample_vectors(543);

        let expected_res = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| x + y)
            .collect::<Vec<f32>>();

        unsafe { f32_xany_avx512_nofma_add_vertical(&mut x, &y) };

        assert_is_close_vector(&x, &expected_res);
    }

    #[test]
    fn test_xany_sub_vertical() {
        let (mut x, y) = get_sample_vectors(543);

        let expected_res = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| x - y)
            .collect::<Vec<f32>>();

        unsafe { f32_xany_avx512_nofma_sub_vertical(&mut x, &y) };

        assert_is_close_vector(&x, &expected_res);
    }
}
