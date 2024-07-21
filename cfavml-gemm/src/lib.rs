#![doc = include_str!("../README.md")]

mod genric_kernel;

use core::arch::x86_64::*;

/// Assumes Row-Major Order.
pub unsafe fn f32_avx2_fma_gemm(
    shape_a: (usize, usize),
    shape_b: (usize, usize),
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    debug_assert_eq!(b.len(), c.len(), "Result matrix size missmatch");
    debug_assert_eq!(a.len(), shape_a.0 * shape_a.1, "Shape error");
    debug_assert_eq!(b.len(), shape_b.0 * shape_b.1, "Shape error");

    let b_ptr = b.as_ptr();
    let c_ptr = c.as_mut_ptr();

    let c_shape = (shape_a.0, shape_b.1);
}
