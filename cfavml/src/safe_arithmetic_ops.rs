//! Vector based arithmetic operations (`add`, `sub`, `mul`, `div`)
//!
//! These exported methods are safe to call and select the fastest available instruction set
//! to use at runtime.
//!
//! Both `xconst` and `xany` variants of each operation are provided.
//!
//! The following arithmetic operations are provided:
//!
//! - Add single value to vector
//! - Sub single value to vector
//! - Mul vector by single value
//! - Div vector by single value
//!   * NOTE: Use multiply by value if you can calculate the inverse for better efficiency.
//!
//! - Add two vectors vertically
//! - Sub two vectors vertically
//! - Mul two vectors vertically
//! - Div two vectors vertically
//!   * NOTE: Non-floating point values likely fall back to scalar operations, not SIMD.
//!
//! # Usage
//!
//!
//! ##### Addition
//!
//! ```
//! use cfavml::*;
//!
//! const DIMS: usize = 3;
//! let a = [1.0, 2.0, 3.0];
//! let b = [1.0, 2.0, 3.0];
//!
//! let mut result_from_value = [0.0f32; DIMS];
//! let mut result_from_vector = [0.0f32; DIMS];
//!
//! f32_xany_add_value(1.0, &a, &mut result_from_value);
//! assert_eq!(result_from_value, [2.0, 3.0, 4.0]);
//!
//! f32_xany_add_vector(&a, &b, &mut result_from_vector);
//! assert_eq!(result_from_vector, [2.0, 4.0, 6.0]);
//! ```
//!
//!
//! ##### Subtraction
//!
//! ```
//! use cfavml::*;
//!
//! const DIMS: usize = 3;
//! let a = [1.0, 2.0, 3.0];
//! let b = [1.0, 2.0, 3.0];
//!
//! let mut result_from_value = [0.0f32; DIMS];
//! let mut result_from_vector = [0.0f32; DIMS];
//!
//! f32_xany_sub_value(1.0, &a, &mut result_from_value);
//! assert_eq!(result_from_value, [0.0, 1.0, 2.0]);
//!
//! f32_xany_sub_vector(&a, &b, &mut result_from_vector);
//! assert_eq!(result_from_vector, [0.0, 0.0, 0.0]);
//! ```
//!
//! ##### Multiplication
//!
//! ```
//! use cfavml::*;
//!
//! const DIMS: usize = 3;
//! let a = [1.0, 2.0, 3.0];
//! let b = [1.0, 2.0, 3.0];
//!
//! let mut result_from_value = [0.0f32; DIMS];
//! let mut result_from_vector = [0.0f32; DIMS];
//!
//! f32_xany_mul_value(2.0, &a, &mut result_from_value);
//! assert_eq!(result_from_value, [2.0, 4.0, 6.0]);
//!
//! f32_xany_mul_vector(&a, &b, &mut result_from_vector);
//! assert_eq!(result_from_vector, [1.0, 4.0, 9.0]);
//! ```
//!
//!
//! ##### Division
//!
//! NOTE:
//! For some things like `f32` and `f64`, you can calculate the inverse of the divisor and instead
//! use a multiply operation. This is significantly faster compute-wise, so if performance is
//! important to you, you should aim to use that approach rather than the `_div_x` operations.
//!
//! ```
//! use cfavml::*;
//!
//! const DIMS: usize = 3;
//! let a = [1.0, 2.0, 3.0];
//! let b = [1.0, 2.0, 3.0];
//!
//! let mut result_from_value = [0.0f32; DIMS];
//! let mut result_from_vector = [0.0f32; DIMS];
//!
//! f32_xany_div_value(2.0, &a, &mut result_from_value);
//! assert_eq!(result_from_value, [0.5, 1.0, 1.5]);
//!
//! f32_xany_div_vector(&a, &b, &mut result_from_vector);
//! assert_eq!(result_from_vector, [1.0, 1.0, 1.0]);
//! ```
use crate::buffer::WriteOnlyBuffer;
use crate::danger::*;

macro_rules! export_safe_arithmetic_vector_x_value_op {
    (
        description = $desc:expr,
        ty = $t:ty,
        const_name = $const_name:ident,
        any_name = $any_name:ident,
        $avx512_const_name:ident,
        $avx2_const_name:ident,
        $neon_const_name:ident,
        $fallback_const_name:ident,
        $avx512_any_name:ident,
        $avx2_any_name:ident,
        $neon_any_name:ident,
        $fallback_any_name:ident,
    ) => {
        #[doc = concat!("`", stringify!($t), "` ", $desc)]
        pub fn $const_name<const DIMS: usize>(
            value: $t,
            a: &[$t],
            result: impl WriteOnlyBuffer<Item = $t>,
        ) {
            assert_eq!(a.len(), DIMS, "Input vector a does not match DIMS");
            assert_eq!(
                result.raw_buffer_len(),
                DIMS,
                "Output vector result does not match DIMS"
            );

            unsafe {
                crate::dispatch!(
                    avx512 = $avx512_const_name::<DIMS>,
                    avx2 = $avx2_const_name::<DIMS>,
                    neon = $neon_const_name::<DIMS>,
                    fallback = $fallback_const_name::<DIMS>,
                    args = (value, a, result)
                );
            }
        }

        #[doc = concat!("`", stringify!($t), "` ", $desc)]
        pub fn $any_name(value: $t, a: &[$t], result: impl WriteOnlyBuffer<Item = $t>) {
            assert_eq!(
                a.len(),
                result.raw_buffer_len(),
                "Input vector and result vector size do not match"
            );

            unsafe {
                crate::dispatch!(
                    avx512 = $avx512_any_name,
                    avx2 = $avx2_any_name,
                    neon = $neon_any_name,
                    fallback = $fallback_any_name,
                    args = (value, a, result)
                );
            }
        }
    };
}

macro_rules! export_safe_arithmetic_vector_x_vector_op {
    (
        description = $desc:expr,
        ty = $t:ty,
        const_name = $const_name:ident,
        any_name = $any_name:ident,
        $avx512_const_name:ident,
        $avx2_const_name:ident,
        $neon_const_name:ident,
        $fallback_const_name:ident,
        $avx512_any_name:ident,
        $avx2_any_name:ident,
        $neon_any_name:ident,
        $fallback_any_name:ident,
    ) => {
        #[doc = concat!("`", stringify!($t), "` ", $desc)]
        pub fn $const_name<const DIMS: usize>(
            a: &[$t],
            b: &[$t],
            result: impl WriteOnlyBuffer<Item = $t>,
        ) {
            assert_eq!(a.len(), DIMS, "Input vector a does not match DIMS");
            assert_eq!(b.len(), DIMS, "Input vector b does not match DIMS");
            assert_eq!(
                a.len(),
                result.raw_buffer_len(),
                "Input vectors and result vector size do not match"
            );

            unsafe {
                crate::dispatch!(
                    avx512 = $avx512_const_name::<DIMS>,
                    avx2 = $avx2_const_name::<DIMS>,
                    neon = $neon_const_name::<DIMS>,
                    fallback = $fallback_const_name::<DIMS>,
                    args = (a, b, result)
                );
            }
        }

        #[doc = concat!("`", stringify!($t), "` ", $desc)]
        pub fn $any_name(a: &[$t], b: &[$t], result: impl WriteOnlyBuffer<Item = $t>) {
            assert_eq!(
                a.len(),
                b.len(),
                "Input vector a and b do not match in size"
            );
            assert_eq!(
                a.len(),
                result.raw_buffer_len(),
                "Input vectors and result vector size do not match"
            );

            unsafe {
                crate::dispatch!(
                    avx512 = $avx512_any_name,
                    avx2 = $avx2_any_name,
                    neon = $neon_any_name,
                    fallback = $fallback_any_name,
                    args = (a, b, result)
                );
            }
        }
    };
}

export_safe_arithmetic_vector_x_value_op!(
    description = "Addition of a single value to `a`, storing the result in `result`",
    ty = f32,
    const_name = f32_xconst_add_value,
    any_name = f32_xany_add_value,
    f32_xconst_avx512_nofma_add_value,
    f32_xconst_avx2_nofma_add_value,
    f32_xconst_neon_nofma_add_value,
    f32_xconst_fallback_nofma_add_value,
    f32_xany_avx512_nofma_add_value,
    f32_xany_avx2_nofma_add_value,
    f32_xany_neon_nofma_add_value,
    f32_xany_fallback_nofma_add_value,
);
export_safe_arithmetic_vector_x_value_op!(
    description =
        "Subtraction of a single value from `a`, storing the result in `result`",
    ty = f32,
    const_name = f32_xconst_sub_value,
    any_name = f32_xany_sub_value,
    f32_xconst_avx512_nofma_sub_value,
    f32_xconst_avx2_nofma_sub_value,
    f32_xconst_neon_nofma_sub_value,
    f32_xconst_fallback_nofma_sub_value,
    f32_xany_avx512_nofma_sub_value,
    f32_xany_avx2_nofma_sub_value,
    f32_xany_neon_nofma_sub_value,
    f32_xany_fallback_nofma_sub_value,
);
export_safe_arithmetic_vector_x_value_op!(
    description = "Multiplication of vector `a` by the value provided, storing the result in `result`",
    ty = f32,
    const_name = f32_xconst_mul_value,
    any_name = f32_xany_mul_value,
    f32_xconst_avx512_nofma_mul_value,
    f32_xconst_avx2_nofma_mul_value,
    f32_xconst_neon_nofma_mul_value,
    f32_xconst_fallback_nofma_mul_value,
    f32_xany_avx512_nofma_mul_value,
    f32_xany_avx2_nofma_mul_value,
    f32_xany_neon_nofma_mul_value,
    f32_xany_fallback_nofma_mul_value,
);
export_safe_arithmetic_vector_x_value_op!(
    description = "\
    Division of vector `a` by the value provided, storing the result in `result`. \
    Note, this method does not do the inverse trick and instead will do the full division operation \
    instead of silently doing a multiply. If your value can calculate the inverse, you should do it
    and use the multiple by value operation instead.\n\n i.e. `f32_xany_mul_value(1.0 / my_value, ...)`
    ",
    ty = f32,
    const_name = f32_xconst_div_value,
    any_name = f32_xany_div_value,
    f32_xconst_avx512_nofma_div_value,
    f32_xconst_avx2_nofma_div_value,
    f32_xconst_neon_nofma_div_value,
    f32_xconst_fallback_nofma_div_value,
    f32_xany_avx512_nofma_div_value,
    f32_xany_avx2_nofma_div_value,
    f32_xany_neon_nofma_div_value,
    f32_xany_fallback_nofma_div_value,
);

export_safe_arithmetic_vector_x_value_op!(
    description = "Addition of a single value to `a`, storing the result in `result`",
    ty = f64,
    const_name = f64_xconst_add_value,
    any_name = f64_xany_add_value,
    f64_xconst_avx512_nofma_add_value,
    f64_xconst_avx2_nofma_add_value,
    f64_xconst_neon_nofma_add_value,
    f64_xconst_fallback_nofma_add_value,
    f64_xany_avx512_nofma_add_value,
    f64_xany_avx2_nofma_add_value,
    f64_xany_neon_nofma_add_value,
    f64_xany_fallback_nofma_add_value,
);
export_safe_arithmetic_vector_x_value_op!(
    description =
        "Subtraction of a single value from `a`, storing the result in `result`",
    ty = f64,
    const_name = f64_xconst_sub_value,
    any_name = f64_xany_sub_value,
    f64_xconst_avx512_nofma_sub_value,
    f64_xconst_avx2_nofma_sub_value,
    f64_xconst_neon_nofma_sub_value,
    f64_xconst_fallback_nofma_sub_value,
    f64_xany_avx512_nofma_sub_value,
    f64_xany_avx2_nofma_sub_value,
    f64_xany_neon_nofma_sub_value,
    f64_xany_fallback_nofma_sub_value,
);
export_safe_arithmetic_vector_x_value_op!(
    description = "Multiplication of vector `a` by the value provided, storing the result in `result`",
    ty = f64,
    const_name = f64_xconst_mul_value,
    any_name = f64_xany_mul_value,
    f64_xconst_avx512_nofma_mul_value,
    f64_xconst_avx2_nofma_mul_value,
    f64_xconst_neon_nofma_mul_value,
    f64_xconst_fallback_nofma_mul_value,
    f64_xany_avx512_nofma_mul_value,
    f64_xany_avx2_nofma_mul_value,
    f64_xany_neon_nofma_mul_value,
    f64_xany_fallback_nofma_mul_value,
);
export_safe_arithmetic_vector_x_value_op!(
    description = "\
    Division of vector `a` by the value provided, storing the result in `result`. \
    Note, this method does not do the inverse trick and instead will do the full division operation \
    instead of silently doing a multiply. If your value can calculate the inverse, you should do it
    and use the multiple by value operation instead.\n\n i.e. `f64_xany_mul_value(1.0 / my_value, ...)`
    ",
    ty = f64,
    const_name = f64_xconst_div_value,
    any_name = f64_xany_div_value,
    f64_xconst_avx512_nofma_div_value,
    f64_xconst_avx2_nofma_div_value,
    f64_xconst_neon_nofma_div_value,
    f64_xconst_fallback_nofma_div_value,
    f64_xany_avx512_nofma_div_value,
    f64_xany_avx2_nofma_div_value,
    f64_xany_neon_nofma_div_value,
    f64_xany_fallback_nofma_div_value,
);

export_safe_arithmetic_vector_x_value_op!(
    description = "Addition of a single value to `a`, storing the result in `result`",
    ty = u8,
    const_name = u8_xconst_add_value,
    any_name = u8_xany_add_value,
    u8_xconst_avx512_nofma_add_value,
    u8_xconst_avx2_nofma_add_value,
    u8_xconst_neon_nofma_add_value,
    u8_xconst_fallback_nofma_add_value,
    u8_xany_avx512_nofma_add_value,
    u8_xany_avx2_nofma_add_value,
    u8_xany_neon_nofma_add_value,
    u8_xany_fallback_nofma_add_value,
);
export_safe_arithmetic_vector_x_value_op!(
    description =
        "Subtraction of a single value from `a`, storing the result in `result`",
    ty = u8,
    const_name = u8_xconst_sub_value,
    any_name = u8_xany_sub_value,
    u8_xconst_avx512_nofma_sub_value,
    u8_xconst_avx2_nofma_sub_value,
    u8_xconst_neon_nofma_sub_value,
    u8_xconst_fallback_nofma_sub_value,
    u8_xany_avx512_nofma_sub_value,
    u8_xany_avx2_nofma_sub_value,
    u8_xany_neon_nofma_sub_value,
    u8_xany_fallback_nofma_sub_value,
);
export_safe_arithmetic_vector_x_value_op!(
    description = "Multiplication of vector `a` by the value provided, storing the result in `result`",
    ty = u8,
    const_name = u8_xconst_mul_value,
    any_name = u8_xany_mul_value,
    u8_xconst_avx512_nofma_mul_value,
    u8_xconst_avx2_nofma_mul_value,
    u8_xconst_neon_nofma_mul_value,
    u8_xconst_fallback_nofma_mul_value,
    u8_xany_avx512_nofma_mul_value,
    u8_xany_avx2_nofma_mul_value,
    u8_xany_neon_nofma_mul_value,
    u8_xany_fallback_nofma_mul_value,
);
export_safe_arithmetic_vector_x_value_op!(
    description =
        "Division of vector `a` by the value provided, storing the result in `result`",
    ty = u8,
    const_name = u8_xconst_div_value,
    any_name = u8_xany_div_value,
    u8_xconst_avx512_nofma_div_value,
    u8_xconst_avx2_nofma_div_value,
    u8_xconst_neon_nofma_div_value,
    u8_xconst_fallback_nofma_div_value,
    u8_xany_avx512_nofma_div_value,
    u8_xany_avx2_nofma_div_value,
    u8_xany_neon_nofma_div_value,
    u8_xany_fallback_nofma_div_value,
);

export_safe_arithmetic_vector_x_value_op!(
    description = "Addition of a single value to `a`, storing the result in `result`",
    ty = u16,
    const_name = u16_xconst_add_value,
    any_name = u16_xany_add_value,
    u16_xconst_avx512_nofma_add_value,
    u16_xconst_avx2_nofma_add_value,
    u16_xconst_neon_nofma_add_value,
    u16_xconst_fallback_nofma_add_value,
    u16_xany_avx512_nofma_add_value,
    u16_xany_avx2_nofma_add_value,
    u16_xany_neon_nofma_add_value,
    u16_xany_fallback_nofma_add_value,
);
export_safe_arithmetic_vector_x_value_op!(
    description =
        "Subtraction of a single value from `a`, storing the result in `result`",
    ty = u16,
    const_name = u16_xconst_sub_value,
    any_name = u16_xany_sub_value,
    u16_xconst_avx512_nofma_sub_value,
    u16_xconst_avx2_nofma_sub_value,
    u16_xconst_neon_nofma_sub_value,
    u16_xconst_fallback_nofma_sub_value,
    u16_xany_avx512_nofma_sub_value,
    u16_xany_avx2_nofma_sub_value,
    u16_xany_neon_nofma_sub_value,
    u16_xany_fallback_nofma_sub_value,
);
export_safe_arithmetic_vector_x_value_op!(
    description = "Multiplication of vector `a` by the value provided, storing the result in `result`",
    ty = u16,
    const_name = u16_xconst_mul_value,
    any_name = u16_xany_mul_value,
    u16_xconst_avx512_nofma_mul_value,
    u16_xconst_avx2_nofma_mul_value,
    u16_xconst_neon_nofma_mul_value,
    u16_xconst_fallback_nofma_mul_value,
    u16_xany_avx512_nofma_mul_value,
    u16_xany_avx2_nofma_mul_value,
    u16_xany_neon_nofma_mul_value,
    u16_xany_fallback_nofma_mul_value,
);
export_safe_arithmetic_vector_x_value_op!(
    description =
        "Division of vector `a` by the value provided, storing the result in `result`",
    ty = u16,
    const_name = u16_xconst_div_value,
    any_name = u16_xany_div_value,
    u16_xconst_avx512_nofma_div_value,
    u16_xconst_avx2_nofma_div_value,
    u16_xconst_neon_nofma_div_value,
    u16_xconst_fallback_nofma_div_value,
    u16_xany_avx512_nofma_div_value,
    u16_xany_avx2_nofma_div_value,
    u16_xany_neon_nofma_div_value,
    u16_xany_fallback_nofma_div_value,
);

export_safe_arithmetic_vector_x_value_op!(
    description = "Addition of a single value to `a`, storing the result in `result`",
    ty = u32,
    const_name = u32_xconst_add_value,
    any_name = u32_xany_add_value,
    u32_xconst_avx512_nofma_add_value,
    u32_xconst_avx2_nofma_add_value,
    u32_xconst_neon_nofma_add_value,
    u32_xconst_fallback_nofma_add_value,
    u32_xany_avx512_nofma_add_value,
    u32_xany_avx2_nofma_add_value,
    u32_xany_neon_nofma_add_value,
    u32_xany_fallback_nofma_add_value,
);
export_safe_arithmetic_vector_x_value_op!(
    description =
        "Subtraction of a single value from `a`, storing the result in `result`",
    ty = u32,
    const_name = u32_xconst_sub_value,
    any_name = u32_xany_sub_value,
    u32_xconst_avx512_nofma_sub_value,
    u32_xconst_avx2_nofma_sub_value,
    u32_xconst_neon_nofma_sub_value,
    u32_xconst_fallback_nofma_sub_value,
    u32_xany_avx512_nofma_sub_value,
    u32_xany_avx2_nofma_sub_value,
    u32_xany_neon_nofma_sub_value,
    u32_xany_fallback_nofma_sub_value,
);
export_safe_arithmetic_vector_x_value_op!(
    description = "Multiplication of vector `a` by the value provided, storing the result in `result`",
    ty = u32,
    const_name = u32_xconst_mul_value,
    any_name = u32_xany_mul_value,
    u32_xconst_avx512_nofma_mul_value,
    u32_xconst_avx2_nofma_mul_value,
    u32_xconst_neon_nofma_mul_value,
    u32_xconst_fallback_nofma_mul_value,
    u32_xany_avx512_nofma_mul_value,
    u32_xany_avx2_nofma_mul_value,
    u32_xany_neon_nofma_mul_value,
    u32_xany_fallback_nofma_mul_value,
);
export_safe_arithmetic_vector_x_value_op!(
    description =
        "Division of vector `a` by the value provided, storing the result in `result`",
    ty = u32,
    const_name = u32_xconst_div_value,
    any_name = u32_xany_div_value,
    u32_xconst_avx512_nofma_div_value,
    u32_xconst_avx2_nofma_div_value,
    u32_xconst_neon_nofma_div_value,
    u32_xconst_fallback_nofma_div_value,
    u32_xany_avx512_nofma_div_value,
    u32_xany_avx2_nofma_div_value,
    u32_xany_neon_nofma_div_value,
    u32_xany_fallback_nofma_div_value,
);

export_safe_arithmetic_vector_x_value_op!(
    description = "Addition of a single value to `a`, storing the result in `result`",
    ty = u64,
    const_name = u64_xconst_add_value,
    any_name = u64_xany_add_value,
    u64_xconst_avx512_nofma_add_value,
    u64_xconst_avx2_nofma_add_value,
    u64_xconst_neon_nofma_add_value,
    u64_xconst_fallback_nofma_add_value,
    u64_xany_avx512_nofma_add_value,
    u64_xany_avx2_nofma_add_value,
    u64_xany_neon_nofma_add_value,
    u64_xany_fallback_nofma_add_value,
);
export_safe_arithmetic_vector_x_value_op!(
    description =
        "Subtraction of a single value from `a`, storing the result in `result`",
    ty = u64,
    const_name = u64_xconst_sub_value,
    any_name = u64_xany_sub_value,
    u64_xconst_avx512_nofma_sub_value,
    u64_xconst_avx2_nofma_sub_value,
    u64_xconst_neon_nofma_sub_value,
    u64_xconst_fallback_nofma_sub_value,
    u64_xany_avx512_nofma_sub_value,
    u64_xany_avx2_nofma_sub_value,
    u64_xany_neon_nofma_sub_value,
    u64_xany_fallback_nofma_sub_value,
);
export_safe_arithmetic_vector_x_value_op!(
    description = "Multiplication of vector `a` by the value provided, storing the result in `result`",
    ty = u64,
    const_name = u64_xconst_mul_value,
    any_name = u64_xany_mul_value,
    u64_xconst_avx512_nofma_mul_value,
    u64_xconst_avx2_nofma_mul_value,
    u64_xconst_neon_nofma_mul_value,
    u64_xconst_fallback_nofma_mul_value,
    u64_xany_avx512_nofma_mul_value,
    u64_xany_avx2_nofma_mul_value,
    u64_xany_neon_nofma_mul_value,
    u64_xany_fallback_nofma_mul_value,
);
export_safe_arithmetic_vector_x_value_op!(
    description =
        "Division of vector `a` by the value provided, storing the result in `result`",
    ty = u64,
    const_name = u64_xconst_div_value,
    any_name = u64_xany_div_value,
    u64_xconst_avx512_nofma_div_value,
    u64_xconst_avx2_nofma_div_value,
    u64_xconst_neon_nofma_div_value,
    u64_xconst_fallback_nofma_div_value,
    u64_xany_avx512_nofma_div_value,
    u64_xany_avx2_nofma_div_value,
    u64_xany_neon_nofma_div_value,
    u64_xany_fallback_nofma_div_value,
);

export_safe_arithmetic_vector_x_value_op!(
    description = "Addition of a single value to `a`, storing the result in `result`",
    ty = i8,
    const_name = i8_xconst_add_value,
    any_name = i8_xany_add_value,
    i8_xconst_avx512_nofma_add_value,
    i8_xconst_avx2_nofma_add_value,
    i8_xconst_neon_nofma_add_value,
    i8_xconst_fallback_nofma_add_value,
    i8_xany_avx512_nofma_add_value,
    i8_xany_avx2_nofma_add_value,
    i8_xany_neon_nofma_add_value,
    i8_xany_fallback_nofma_add_value,
);
export_safe_arithmetic_vector_x_value_op!(
    description =
        "Subtraction of a single value from `a`, storing the result in `result`",
    ty = i8,
    const_name = i8_xconst_sub_value,
    any_name = i8_xany_sub_value,
    i8_xconst_avx512_nofma_sub_value,
    i8_xconst_avx2_nofma_sub_value,
    i8_xconst_neon_nofma_sub_value,
    i8_xconst_fallback_nofma_sub_value,
    i8_xany_avx512_nofma_sub_value,
    i8_xany_avx2_nofma_sub_value,
    i8_xany_neon_nofma_sub_value,
    i8_xany_fallback_nofma_sub_value,
);
export_safe_arithmetic_vector_x_value_op!(
    description = "Multiplication of vector `a` by the value provided, storing the result in `result`",
    ty = i8,
    const_name = i8_xconst_mul_value,
    any_name = i8_xany_mul_value,
    i8_xconst_avx512_nofma_mul_value,
    i8_xconst_avx2_nofma_mul_value,
    i8_xconst_neon_nofma_mul_value,
    i8_xconst_fallback_nofma_mul_value,
    i8_xany_avx512_nofma_mul_value,
    i8_xany_avx2_nofma_mul_value,
    i8_xany_neon_nofma_mul_value,
    i8_xany_fallback_nofma_mul_value,
);
export_safe_arithmetic_vector_x_value_op!(
    description =
        "Division of vector `a` by the value provided, storing the result in `result`",
    ty = i8,
    const_name = i8_xconst_div_value,
    any_name = i8_xany_div_value,
    i8_xconst_avx512_nofma_div_value,
    i8_xconst_avx2_nofma_div_value,
    i8_xconst_neon_nofma_div_value,
    i8_xconst_fallback_nofma_div_value,
    i8_xany_avx512_nofma_div_value,
    i8_xany_avx2_nofma_div_value,
    i8_xany_neon_nofma_div_value,
    i8_xany_fallback_nofma_div_value,
);

export_safe_arithmetic_vector_x_value_op!(
    description = "Addition of a single value to `a`, storing the result in `result`",
    ty = i16,
    const_name = i16_xconst_add_value,
    any_name = i16_xany_add_value,
    i16_xconst_avx512_nofma_add_value,
    i16_xconst_avx2_nofma_add_value,
    i16_xconst_neon_nofma_add_value,
    i16_xconst_fallback_nofma_add_value,
    i16_xany_avx512_nofma_add_value,
    i16_xany_avx2_nofma_add_value,
    i16_xany_neon_nofma_add_value,
    i16_xany_fallback_nofma_add_value,
);
export_safe_arithmetic_vector_x_value_op!(
    description =
        "Subtraction of a single value from `a`, storing the result in `result`",
    ty = i16,
    const_name = i16_xconst_sub_value,
    any_name = i16_xany_sub_value,
    i16_xconst_avx512_nofma_sub_value,
    i16_xconst_avx2_nofma_sub_value,
    i16_xconst_neon_nofma_sub_value,
    i16_xconst_fallback_nofma_sub_value,
    i16_xany_avx512_nofma_sub_value,
    i16_xany_avx2_nofma_sub_value,
    i16_xany_neon_nofma_sub_value,
    i16_xany_fallback_nofma_sub_value,
);
export_safe_arithmetic_vector_x_value_op!(
    description = "Multiplication of vector `a` by the value provided, storing the result in `result`",
    ty = i16,
    const_name = i16_xconst_mul_value,
    any_name = i16_xany_mul_value,
    i16_xconst_avx512_nofma_mul_value,
    i16_xconst_avx2_nofma_mul_value,
    i16_xconst_neon_nofma_mul_value,
    i16_xconst_fallback_nofma_mul_value,
    i16_xany_avx512_nofma_mul_value,
    i16_xany_avx2_nofma_mul_value,
    i16_xany_neon_nofma_mul_value,
    i16_xany_fallback_nofma_mul_value,
);
export_safe_arithmetic_vector_x_value_op!(
    description =
        "Division of vector `a` by the value provided, storing the result in `result`",
    ty = i16,
    const_name = i16_xconst_div_value,
    any_name = i16_xany_div_value,
    i16_xconst_avx512_nofma_div_value,
    i16_xconst_avx2_nofma_div_value,
    i16_xconst_neon_nofma_div_value,
    i16_xconst_fallback_nofma_div_value,
    i16_xany_avx512_nofma_div_value,
    i16_xany_avx2_nofma_div_value,
    i16_xany_neon_nofma_div_value,
    i16_xany_fallback_nofma_div_value,
);

export_safe_arithmetic_vector_x_value_op!(
    description = "Addition of a single value to `a`, storing the result in `result`",
    ty = i32,
    const_name = i32_xconst_add_value,
    any_name = i32_xany_add_value,
    i32_xconst_avx512_nofma_add_value,
    i32_xconst_avx2_nofma_add_value,
    i32_xconst_neon_nofma_add_value,
    i32_xconst_fallback_nofma_add_value,
    i32_xany_avx512_nofma_add_value,
    i32_xany_avx2_nofma_add_value,
    i32_xany_neon_nofma_add_value,
    i32_xany_fallback_nofma_add_value,
);
export_safe_arithmetic_vector_x_value_op!(
    description =
        "Subtraction of a single value from `a`, storing the result in `result`",
    ty = i32,
    const_name = i32_xconst_sub_value,
    any_name = i32_xany_sub_value,
    i32_xconst_avx512_nofma_sub_value,
    i32_xconst_avx2_nofma_sub_value,
    i32_xconst_neon_nofma_sub_value,
    i32_xconst_fallback_nofma_sub_value,
    i32_xany_avx512_nofma_sub_value,
    i32_xany_avx2_nofma_sub_value,
    i32_xany_neon_nofma_sub_value,
    i32_xany_fallback_nofma_sub_value,
);
export_safe_arithmetic_vector_x_value_op!(
    description = "Multiplication of vector `a` by the value provided, storing the result in `result`",
    ty = i32,
    const_name = i32_xconst_mul_value,
    any_name = i32_xany_mul_value,
    i32_xconst_avx512_nofma_mul_value,
    i32_xconst_avx2_nofma_mul_value,
    i32_xconst_neon_nofma_mul_value,
    i32_xconst_fallback_nofma_mul_value,
    i32_xany_avx512_nofma_mul_value,
    i32_xany_avx2_nofma_mul_value,
    i32_xany_neon_nofma_mul_value,
    i32_xany_fallback_nofma_mul_value,
);
export_safe_arithmetic_vector_x_value_op!(
    description =
        "Division of vector `a` by the value provided, storing the result in `result`",
    ty = i32,
    const_name = i32_xconst_div_value,
    any_name = i32_xany_div_value,
    i32_xconst_avx512_nofma_div_value,
    i32_xconst_avx2_nofma_div_value,
    i32_xconst_neon_nofma_div_value,
    i32_xconst_fallback_nofma_div_value,
    i32_xany_avx512_nofma_div_value,
    i32_xany_avx2_nofma_div_value,
    i32_xany_neon_nofma_div_value,
    i32_xany_fallback_nofma_div_value,
);

export_safe_arithmetic_vector_x_value_op!(
    description = "Addition of a single value to `a`, storing the result in `result`",
    ty = i64,
    const_name = i64_xconst_add_value,
    any_name = i64_xany_add_value,
    i64_xconst_avx512_nofma_add_value,
    i64_xconst_avx2_nofma_add_value,
    i64_xconst_neon_nofma_add_value,
    i64_xconst_fallback_nofma_add_value,
    i64_xany_avx512_nofma_add_value,
    i64_xany_avx2_nofma_add_value,
    i64_xany_neon_nofma_add_value,
    i64_xany_fallback_nofma_add_value,
);
export_safe_arithmetic_vector_x_value_op!(
    description =
        "Subtraction of a single value from `a`, storing the result in `result`",
    ty = i64,
    const_name = i64_xconst_sub_value,
    any_name = i64_xany_sub_value,
    i64_xconst_avx512_nofma_sub_value,
    i64_xconst_avx2_nofma_sub_value,
    i64_xconst_neon_nofma_sub_value,
    i64_xconst_fallback_nofma_sub_value,
    i64_xany_avx512_nofma_sub_value,
    i64_xany_avx2_nofma_sub_value,
    i64_xany_neon_nofma_sub_value,
    i64_xany_fallback_nofma_sub_value,
);
export_safe_arithmetic_vector_x_value_op!(
    description = "Multiplication of vector `a` by the value provided, storing the result in `result`",
    ty = i64,
    const_name = i64_xconst_mul_value,
    any_name = i64_xany_mul_value,
    i64_xconst_avx512_nofma_mul_value,
    i64_xconst_avx2_nofma_mul_value,
    i64_xconst_neon_nofma_mul_value,
    i64_xconst_fallback_nofma_mul_value,
    i64_xany_avx512_nofma_mul_value,
    i64_xany_avx2_nofma_mul_value,
    i64_xany_neon_nofma_mul_value,
    i64_xany_fallback_nofma_mul_value,
);
export_safe_arithmetic_vector_x_value_op!(
    description =
        "Division of vector `a` by the value provided, storing the result in `result`",
    ty = i64,
    const_name = i64_xconst_div_value,
    any_name = i64_xany_div_value,
    i64_xconst_avx512_nofma_div_value,
    i64_xconst_avx2_nofma_div_value,
    i64_xconst_neon_nofma_div_value,
    i64_xconst_fallback_nofma_div_value,
    i64_xany_avx512_nofma_div_value,
    i64_xany_avx2_nofma_div_value,
    i64_xany_neon_nofma_div_value,
    i64_xany_fallback_nofma_div_value,
);

export_safe_arithmetic_vector_x_vector_op!(
    description = "Addition of vector `a` and `b`, storing the result in `result`",
    ty = f32,
    const_name = f32_xconst_add_vector,
    any_name = f32_xany_add_vector,
    f32_xconst_avx512_nofma_add_vector,
    f32_xconst_avx2_nofma_add_vector,
    f32_xconst_neon_nofma_add_vector,
    f32_xconst_fallback_nofma_add_vector,
    f32_xany_avx512_nofma_add_vector,
    f32_xany_avx2_nofma_add_vector,
    f32_xany_neon_nofma_add_vector,
    f32_xany_fallback_nofma_add_vector,
);
export_safe_arithmetic_vector_x_vector_op!(
    description = "Subtraction of vector `b` from `a`, storing the result in `result`",
    ty = f32,
    const_name = f32_xconst_sub_vector,
    any_name = f32_xany_sub_vector,
    f32_xconst_avx512_nofma_sub_vector,
    f32_xconst_avx2_nofma_sub_vector,
    f32_xconst_neon_nofma_sub_vector,
    f32_xconst_fallback_nofma_sub_vector,
    f32_xany_avx512_nofma_sub_vector,
    f32_xany_avx2_nofma_sub_vector,
    f32_xany_neon_nofma_sub_vector,
    f32_xany_fallback_nofma_sub_vector,
);
export_safe_arithmetic_vector_x_vector_op!(
    description = "Multiplication of vector `a` by `b, storing the result in `result`",
    ty = f32,
    const_name = f32_xconst_mul_vector,
    any_name = f32_xany_mul_vector,
    f32_xconst_avx512_nofma_mul_vector,
    f32_xconst_avx2_nofma_mul_vector,
    f32_xconst_neon_nofma_mul_vector,
    f32_xconst_fallback_nofma_mul_vector,
    f32_xany_avx512_nofma_mul_vector,
    f32_xany_avx2_nofma_mul_vector,
    f32_xany_neon_nofma_mul_vector,
    f32_xany_fallback_nofma_mul_vector,
);
export_safe_arithmetic_vector_x_vector_op!(
    description = "Division of vector `a` by vector `b`, storing the result in `result`",
    ty = f32,
    const_name = f32_xconst_div_vector,
    any_name = f32_xany_div_vector,
    f32_xconst_avx512_nofma_div_vector,
    f32_xconst_avx2_nofma_div_vector,
    f32_xconst_neon_nofma_div_vector,
    f32_xconst_fallback_nofma_div_vector,
    f32_xany_avx512_nofma_div_vector,
    f32_xany_avx2_nofma_div_vector,
    f32_xany_neon_nofma_div_vector,
    f32_xany_fallback_nofma_div_vector,
);

export_safe_arithmetic_vector_x_vector_op!(
    description = "Addition of vector `a` and `b`, storing the result in `result`",
    ty = f64,
    const_name = f64_xconst_add_vector,
    any_name = f64_xany_add_vector,
    f64_xconst_avx512_nofma_add_vector,
    f64_xconst_avx2_nofma_add_vector,
    f64_xconst_neon_nofma_add_vector,
    f64_xconst_fallback_nofma_add_vector,
    f64_xany_avx512_nofma_add_vector,
    f64_xany_avx2_nofma_add_vector,
    f64_xany_neon_nofma_add_vector,
    f64_xany_fallback_nofma_add_vector,
);
export_safe_arithmetic_vector_x_vector_op!(
    description = "Subtraction of vector `b` from `a`, storing the result in `result`",
    ty = f64,
    const_name = f64_xconst_sub_vector,
    any_name = f64_xany_sub_vector,
    f64_xconst_avx512_nofma_sub_vector,
    f64_xconst_avx2_nofma_sub_vector,
    f64_xconst_neon_nofma_sub_vector,
    f64_xconst_fallback_nofma_sub_vector,
    f64_xany_avx512_nofma_sub_vector,
    f64_xany_avx2_nofma_sub_vector,
    f64_xany_neon_nofma_sub_vector,
    f64_xany_fallback_nofma_sub_vector,
);
export_safe_arithmetic_vector_x_vector_op!(
    description = "Multiplication of vector `a` by `b, storing the result in `result`",
    ty = f64,
    const_name = f64_xconst_mul_vector,
    any_name = f64_xany_mul_vector,
    f64_xconst_avx512_nofma_mul_vector,
    f64_xconst_avx2_nofma_mul_vector,
    f64_xconst_neon_nofma_mul_vector,
    f64_xconst_fallback_nofma_mul_vector,
    f64_xany_avx512_nofma_mul_vector,
    f64_xany_avx2_nofma_mul_vector,
    f64_xany_neon_nofma_mul_vector,
    f64_xany_fallback_nofma_mul_vector,
);
export_safe_arithmetic_vector_x_vector_op!(
    description = "Division of vector `a` by vector `b`, storing the result in `result`",
    ty = f64,
    const_name = f64_xconst_div_vector,
    any_name = f64_xany_div_vector,
    f64_xconst_avx512_nofma_div_vector,
    f64_xconst_avx2_nofma_div_vector,
    f64_xconst_neon_nofma_div_vector,
    f64_xconst_fallback_nofma_div_vector,
    f64_xany_avx512_nofma_div_vector,
    f64_xany_avx2_nofma_div_vector,
    f64_xany_neon_nofma_div_vector,
    f64_xany_fallback_nofma_div_vector,
);

export_safe_arithmetic_vector_x_vector_op!(
    description = "Addition of vector `a` and `b`, storing the result in `result`",
    ty = u8,
    const_name = u8_xconst_add_vector,
    any_name = u8_xany_add_vector,
    u8_xconst_avx512_nofma_add_vector,
    u8_xconst_avx2_nofma_add_vector,
    u8_xconst_neon_nofma_add_vector,
    u8_xconst_fallback_nofma_add_vector,
    u8_xany_avx512_nofma_add_vector,
    u8_xany_avx2_nofma_add_vector,
    u8_xany_neon_nofma_add_vector,
    u8_xany_fallback_nofma_add_vector,
);
export_safe_arithmetic_vector_x_vector_op!(
    description = "Subtraction of vector `b` from `a`, storing the result in `result`",
    ty = u8,
    const_name = u8_xconst_sub_vector,
    any_name = u8_xany_sub_vector,
    u8_xconst_avx512_nofma_sub_vector,
    u8_xconst_avx2_nofma_sub_vector,
    u8_xconst_neon_nofma_sub_vector,
    u8_xconst_fallback_nofma_sub_vector,
    u8_xany_avx512_nofma_sub_vector,
    u8_xany_avx2_nofma_sub_vector,
    u8_xany_neon_nofma_sub_vector,
    u8_xany_fallback_nofma_sub_vector,
);
export_safe_arithmetic_vector_x_vector_op!(
    description = "Multiplication of vector `a` by `b, storing the result in `result`",
    ty = u8,
    const_name = u8_xconst_mul_vector,
    any_name = u8_xany_mul_vector,
    u8_xconst_avx512_nofma_mul_vector,
    u8_xconst_avx2_nofma_mul_vector,
    u8_xconst_neon_nofma_mul_vector,
    u8_xconst_fallback_nofma_mul_vector,
    u8_xany_avx512_nofma_mul_vector,
    u8_xany_avx2_nofma_mul_vector,
    u8_xany_neon_nofma_mul_vector,
    u8_xany_fallback_nofma_mul_vector,
);
export_safe_arithmetic_vector_x_vector_op!(
    description = "Division of vector `a` by vector `b`, storing the result in `result`",
    ty = u8,
    const_name = u8_xconst_div_vector,
    any_name = u8_xany_div_vector,
    u8_xconst_avx512_nofma_div_vector,
    u8_xconst_avx2_nofma_div_vector,
    u8_xconst_neon_nofma_div_vector,
    u8_xconst_fallback_nofma_div_vector,
    u8_xany_avx512_nofma_div_vector,
    u8_xany_avx2_nofma_div_vector,
    u8_xany_neon_nofma_div_vector,
    u8_xany_fallback_nofma_div_vector,
);

export_safe_arithmetic_vector_x_vector_op!(
    description = "Addition of vector `a` and `b`, storing the result in `result`",
    ty = u16,
    const_name = u16_xconst_add_vector,
    any_name = u16_xany_add_vector,
    u16_xconst_avx512_nofma_add_vector,
    u16_xconst_avx2_nofma_add_vector,
    u16_xconst_neon_nofma_add_vector,
    u16_xconst_fallback_nofma_add_vector,
    u16_xany_avx512_nofma_add_vector,
    u16_xany_avx2_nofma_add_vector,
    u16_xany_neon_nofma_add_vector,
    u16_xany_fallback_nofma_add_vector,
);
export_safe_arithmetic_vector_x_vector_op!(
    description = "Subtraction of vector `b` from `a`, storing the result in `result`",
    ty = u16,
    const_name = u16_xconst_sub_vector,
    any_name = u16_xany_sub_vector,
    u16_xconst_avx512_nofma_sub_vector,
    u16_xconst_avx2_nofma_sub_vector,
    u16_xconst_neon_nofma_sub_vector,
    u16_xconst_fallback_nofma_sub_vector,
    u16_xany_avx512_nofma_sub_vector,
    u16_xany_avx2_nofma_sub_vector,
    u16_xany_neon_nofma_sub_vector,
    u16_xany_fallback_nofma_sub_vector,
);
export_safe_arithmetic_vector_x_vector_op!(
    description = "Multiplication of vector `a` by `b, storing the result in `result`",
    ty = u16,
    const_name = u16_xconst_mul_vector,
    any_name = u16_xany_mul_vector,
    u16_xconst_avx512_nofma_mul_vector,
    u16_xconst_avx2_nofma_mul_vector,
    u16_xconst_neon_nofma_mul_vector,
    u16_xconst_fallback_nofma_mul_vector,
    u16_xany_avx512_nofma_mul_vector,
    u16_xany_avx2_nofma_mul_vector,
    u16_xany_neon_nofma_mul_vector,
    u16_xany_fallback_nofma_mul_vector,
);
export_safe_arithmetic_vector_x_vector_op!(
    description = "Division of vector `a` by vector `b`, storing the result in `result`",
    ty = u16,
    const_name = u16_xconst_div_vector,
    any_name = u16_xany_div_vector,
    u16_xconst_avx512_nofma_div_vector,
    u16_xconst_avx2_nofma_div_vector,
    u16_xconst_neon_nofma_div_vector,
    u16_xconst_fallback_nofma_div_vector,
    u16_xany_avx512_nofma_div_vector,
    u16_xany_avx2_nofma_div_vector,
    u16_xany_neon_nofma_div_vector,
    u16_xany_fallback_nofma_div_vector,
);

export_safe_arithmetic_vector_x_vector_op!(
    description = "Addition of vector `a` and `b`, storing the result in `result`",
    ty = u32,
    const_name = u32_xconst_add_vector,
    any_name = u32_xany_add_vector,
    u32_xconst_avx512_nofma_add_vector,
    u32_xconst_avx2_nofma_add_vector,
    u32_xconst_neon_nofma_add_vector,
    u32_xconst_fallback_nofma_add_vector,
    u32_xany_avx512_nofma_add_vector,
    u32_xany_avx2_nofma_add_vector,
    u32_xany_neon_nofma_add_vector,
    u32_xany_fallback_nofma_add_vector,
);
export_safe_arithmetic_vector_x_vector_op!(
    description = "Subtraction of vector `b` from `a`, storing the result in `result`",
    ty = u32,
    const_name = u32_xconst_sub_vector,
    any_name = u32_xany_sub_vector,
    u32_xconst_avx512_nofma_sub_vector,
    u32_xconst_avx2_nofma_sub_vector,
    u32_xconst_neon_nofma_sub_vector,
    u32_xconst_fallback_nofma_sub_vector,
    u32_xany_avx512_nofma_sub_vector,
    u32_xany_avx2_nofma_sub_vector,
    u32_xany_neon_nofma_sub_vector,
    u32_xany_fallback_nofma_sub_vector,
);
export_safe_arithmetic_vector_x_vector_op!(
    description = "Multiplication of vector `a` by `b, storing the result in `result`",
    ty = u32,
    const_name = u32_xconst_mul_vector,
    any_name = u32_xany_mul_vector,
    u32_xconst_avx512_nofma_mul_vector,
    u32_xconst_avx2_nofma_mul_vector,
    u32_xconst_neon_nofma_mul_vector,
    u32_xconst_fallback_nofma_mul_vector,
    u32_xany_avx512_nofma_mul_vector,
    u32_xany_avx2_nofma_mul_vector,
    u32_xany_neon_nofma_mul_vector,
    u32_xany_fallback_nofma_mul_vector,
);
export_safe_arithmetic_vector_x_vector_op!(
    description = "Division of vector `a` by vector `b`, storing the result in `result`",
    ty = u32,
    const_name = u32_xconst_div_vector,
    any_name = u32_xany_div_vector,
    u32_xconst_avx512_nofma_div_vector,
    u32_xconst_avx2_nofma_div_vector,
    u32_xconst_neon_nofma_div_vector,
    u32_xconst_fallback_nofma_div_vector,
    u32_xany_avx512_nofma_div_vector,
    u32_xany_avx2_nofma_div_vector,
    u32_xany_neon_nofma_div_vector,
    u32_xany_fallback_nofma_div_vector,
);

export_safe_arithmetic_vector_x_vector_op!(
    description = "Addition of vector `a` and `b`, storing the result in `result`",
    ty = u64,
    const_name = u64_xconst_add_vector,
    any_name = u64_xany_add_vector,
    u64_xconst_avx512_nofma_add_vector,
    u64_xconst_avx2_nofma_add_vector,
    u64_xconst_neon_nofma_add_vector,
    u64_xconst_fallback_nofma_add_vector,
    u64_xany_avx512_nofma_add_vector,
    u64_xany_avx2_nofma_add_vector,
    u64_xany_neon_nofma_add_vector,
    u64_xany_fallback_nofma_add_vector,
);
export_safe_arithmetic_vector_x_vector_op!(
    description = "Subtraction of vector `b` from `a`, storing the result in `result`",
    ty = u64,
    const_name = u64_xconst_sub_vector,
    any_name = u64_xany_sub_vector,
    u64_xconst_avx512_nofma_sub_vector,
    u64_xconst_avx2_nofma_sub_vector,
    u64_xconst_neon_nofma_sub_vector,
    u64_xconst_fallback_nofma_sub_vector,
    u64_xany_avx512_nofma_sub_vector,
    u64_xany_avx2_nofma_sub_vector,
    u64_xany_neon_nofma_sub_vector,
    u64_xany_fallback_nofma_sub_vector,
);
export_safe_arithmetic_vector_x_vector_op!(
    description = "Multiplication of vector `a` by `b, storing the result in `result`",
    ty = u64,
    const_name = u64_xconst_mul_vector,
    any_name = u64_xany_mul_vector,
    u64_xconst_avx512_nofma_mul_vector,
    u64_xconst_avx2_nofma_mul_vector,
    u64_xconst_neon_nofma_mul_vector,
    u64_xconst_fallback_nofma_mul_vector,
    u64_xany_avx512_nofma_mul_vector,
    u64_xany_avx2_nofma_mul_vector,
    u64_xany_neon_nofma_mul_vector,
    u64_xany_fallback_nofma_mul_vector,
);
export_safe_arithmetic_vector_x_vector_op!(
    description = "Division of vector `a` by vector `b`, storing the result in `result`",
    ty = u64,
    const_name = u64_xconst_div_vector,
    any_name = u64_xany_div_vector,
    u64_xconst_avx512_nofma_div_vector,
    u64_xconst_avx2_nofma_div_vector,
    u64_xconst_neon_nofma_div_vector,
    u64_xconst_fallback_nofma_div_vector,
    u64_xany_avx512_nofma_div_vector,
    u64_xany_avx2_nofma_div_vector,
    u64_xany_neon_nofma_div_vector,
    u64_xany_fallback_nofma_div_vector,
);

export_safe_arithmetic_vector_x_vector_op!(
    description = "Addition of vector `a` and `b`, storing the result in `result`",
    ty = i8,
    const_name = i8_xconst_add_vector,
    any_name = i8_xany_add_vector,
    i8_xconst_avx512_nofma_add_vector,
    i8_xconst_avx2_nofma_add_vector,
    i8_xconst_neon_nofma_add_vector,
    i8_xconst_fallback_nofma_add_vector,
    i8_xany_avx512_nofma_add_vector,
    i8_xany_avx2_nofma_add_vector,
    i8_xany_neon_nofma_add_vector,
    i8_xany_fallback_nofma_add_vector,
);
export_safe_arithmetic_vector_x_vector_op!(
    description = "Subtraction of vector `b` from `a`, storing the result in `result`",
    ty = i8,
    const_name = i8_xconst_sub_vector,
    any_name = i8_xany_sub_vector,
    i8_xconst_avx512_nofma_sub_vector,
    i8_xconst_avx2_nofma_sub_vector,
    i8_xconst_neon_nofma_sub_vector,
    i8_xconst_fallback_nofma_sub_vector,
    i8_xany_avx512_nofma_sub_vector,
    i8_xany_avx2_nofma_sub_vector,
    i8_xany_neon_nofma_sub_vector,
    i8_xany_fallback_nofma_sub_vector,
);
export_safe_arithmetic_vector_x_vector_op!(
    description = "Multiplication of vector `a` by `b, storing the result in `result`",
    ty = i8,
    const_name = i8_xconst_mul_vector,
    any_name = i8_xany_mul_vector,
    i8_xconst_avx512_nofma_mul_vector,
    i8_xconst_avx2_nofma_mul_vector,
    i8_xconst_neon_nofma_mul_vector,
    i8_xconst_fallback_nofma_mul_vector,
    i8_xany_avx512_nofma_mul_vector,
    i8_xany_avx2_nofma_mul_vector,
    i8_xany_neon_nofma_mul_vector,
    i8_xany_fallback_nofma_mul_vector,
);
export_safe_arithmetic_vector_x_vector_op!(
    description = "Division of vector `a` by vector `b`, storing the result in `result`",
    ty = i8,
    const_name = i8_xconst_div_vector,
    any_name = i8_xany_div_vector,
    i8_xconst_avx512_nofma_div_vector,
    i8_xconst_avx2_nofma_div_vector,
    i8_xconst_neon_nofma_div_vector,
    i8_xconst_fallback_nofma_div_vector,
    i8_xany_avx512_nofma_div_vector,
    i8_xany_avx2_nofma_div_vector,
    i8_xany_neon_nofma_div_vector,
    i8_xany_fallback_nofma_div_vector,
);

export_safe_arithmetic_vector_x_vector_op!(
    description = "Addition of vector `a` and `b`, storing the result in `result`",
    ty = i16,
    const_name = i16_xconst_add_vector,
    any_name = i16_xany_add_vector,
    i16_xconst_avx512_nofma_add_vector,
    i16_xconst_avx2_nofma_add_vector,
    i16_xconst_neon_nofma_add_vector,
    i16_xconst_fallback_nofma_add_vector,
    i16_xany_avx512_nofma_add_vector,
    i16_xany_avx2_nofma_add_vector,
    i16_xany_neon_nofma_add_vector,
    i16_xany_fallback_nofma_add_vector,
);
export_safe_arithmetic_vector_x_vector_op!(
    description = "Subtraction of vector `b` from `a`, storing the result in `result`",
    ty = i16,
    const_name = i16_xconst_sub_vector,
    any_name = i16_xany_sub_vector,
    i16_xconst_avx512_nofma_sub_vector,
    i16_xconst_avx2_nofma_sub_vector,
    i16_xconst_neon_nofma_sub_vector,
    i16_xconst_fallback_nofma_sub_vector,
    i16_xany_avx512_nofma_sub_vector,
    i16_xany_avx2_nofma_sub_vector,
    i16_xany_neon_nofma_sub_vector,
    i16_xany_fallback_nofma_sub_vector,
);
export_safe_arithmetic_vector_x_vector_op!(
    description = "Multiplication of vector `a` by `b, storing the result in `result`",
    ty = i16,
    const_name = i16_xconst_mul_vector,
    any_name = i16_xany_mul_vector,
    i16_xconst_avx512_nofma_mul_vector,
    i16_xconst_avx2_nofma_mul_vector,
    i16_xconst_neon_nofma_mul_vector,
    i16_xconst_fallback_nofma_mul_vector,
    i16_xany_avx512_nofma_mul_vector,
    i16_xany_avx2_nofma_mul_vector,
    i16_xany_neon_nofma_mul_vector,
    i16_xany_fallback_nofma_mul_vector,
);
export_safe_arithmetic_vector_x_vector_op!(
    description = "Division of vector `a` by vector `b`, storing the result in `result`",
    ty = i16,
    const_name = i16_xconst_div_vector,
    any_name = i16_xany_div_vector,
    i16_xconst_avx512_nofma_div_vector,
    i16_xconst_avx2_nofma_div_vector,
    i16_xconst_neon_nofma_div_vector,
    i16_xconst_fallback_nofma_div_vector,
    i16_xany_avx512_nofma_div_vector,
    i16_xany_avx2_nofma_div_vector,
    i16_xany_neon_nofma_div_vector,
    i16_xany_fallback_nofma_div_vector,
);

export_safe_arithmetic_vector_x_vector_op!(
    description = "Addition of vector `a` and `b`, storing the result in `result`",
    ty = i32,
    const_name = i32_xconst_add_vector,
    any_name = i32_xany_add_vector,
    i32_xconst_avx512_nofma_add_vector,
    i32_xconst_avx2_nofma_add_vector,
    i32_xconst_neon_nofma_add_vector,
    i32_xconst_fallback_nofma_add_vector,
    i32_xany_avx512_nofma_add_vector,
    i32_xany_avx2_nofma_add_vector,
    i32_xany_neon_nofma_add_vector,
    i32_xany_fallback_nofma_add_vector,
);
export_safe_arithmetic_vector_x_vector_op!(
    description = "Subtraction of vector `b` from `a`, storing the result in `result`",
    ty = i32,
    const_name = i32_xconst_sub_vector,
    any_name = i32_xany_sub_vector,
    i32_xconst_avx512_nofma_sub_vector,
    i32_xconst_avx2_nofma_sub_vector,
    i32_xconst_neon_nofma_sub_vector,
    i32_xconst_fallback_nofma_sub_vector,
    i32_xany_avx512_nofma_sub_vector,
    i32_xany_avx2_nofma_sub_vector,
    i32_xany_neon_nofma_sub_vector,
    i32_xany_fallback_nofma_sub_vector,
);
export_safe_arithmetic_vector_x_vector_op!(
    description = "Multiplication of vector `a` by `b, storing the result in `result`",
    ty = i32,
    const_name = i32_xconst_mul_vector,
    any_name = i32_xany_mul_vector,
    i32_xconst_avx512_nofma_mul_vector,
    i32_xconst_avx2_nofma_mul_vector,
    i32_xconst_neon_nofma_mul_vector,
    i32_xconst_fallback_nofma_mul_vector,
    i32_xany_avx512_nofma_mul_vector,
    i32_xany_avx2_nofma_mul_vector,
    i32_xany_neon_nofma_mul_vector,
    i32_xany_fallback_nofma_mul_vector,
);
export_safe_arithmetic_vector_x_vector_op!(
    description = "Division of vector `a` by vector `b`, storing the result in `result`",
    ty = i32,
    const_name = i32_xconst_div_vector,
    any_name = i32_xany_div_vector,
    i32_xconst_avx512_nofma_div_vector,
    i32_xconst_avx2_nofma_div_vector,
    i32_xconst_neon_nofma_div_vector,
    i32_xconst_fallback_nofma_div_vector,
    i32_xany_avx512_nofma_div_vector,
    i32_xany_avx2_nofma_div_vector,
    i32_xany_neon_nofma_div_vector,
    i32_xany_fallback_nofma_div_vector,
);

export_safe_arithmetic_vector_x_vector_op!(
    description = "Addition of vector `a` and `b`, storing the result in `result`",
    ty = i64,
    const_name = i64_xconst_add_vector,
    any_name = i64_xany_add_vector,
    i64_xconst_avx512_nofma_add_vector,
    i64_xconst_avx2_nofma_add_vector,
    i64_xconst_neon_nofma_add_vector,
    i64_xconst_fallback_nofma_add_vector,
    i64_xany_avx512_nofma_add_vector,
    i64_xany_avx2_nofma_add_vector,
    i64_xany_neon_nofma_add_vector,
    i64_xany_fallback_nofma_add_vector,
);
export_safe_arithmetic_vector_x_vector_op!(
    description = "Subtraction of vector `b` from `a`, storing the result in `result`",
    ty = i64,
    const_name = i64_xconst_sub_vector,
    any_name = i64_xany_sub_vector,
    i64_xconst_avx512_nofma_sub_vector,
    i64_xconst_avx2_nofma_sub_vector,
    i64_xconst_neon_nofma_sub_vector,
    i64_xconst_fallback_nofma_sub_vector,
    i64_xany_avx512_nofma_sub_vector,
    i64_xany_avx2_nofma_sub_vector,
    i64_xany_neon_nofma_sub_vector,
    i64_xany_fallback_nofma_sub_vector,
);
export_safe_arithmetic_vector_x_vector_op!(
    description = "Multiplication of vector `a` by `b, storing the result in `result`",
    ty = i64,
    const_name = i64_xconst_mul_vector,
    any_name = i64_xany_mul_vector,
    i64_xconst_avx512_nofma_mul_vector,
    i64_xconst_avx2_nofma_mul_vector,
    i64_xconst_neon_nofma_mul_vector,
    i64_xconst_fallback_nofma_mul_vector,
    i64_xany_avx512_nofma_mul_vector,
    i64_xany_avx2_nofma_mul_vector,
    i64_xany_neon_nofma_mul_vector,
    i64_xany_fallback_nofma_mul_vector,
);
export_safe_arithmetic_vector_x_vector_op!(
    description = "Division of vector `a` by vector `b`, storing the result in `result`",
    ty = i64,
    const_name = i64_xconst_div_vector,
    any_name = i64_xany_div_vector,
    i64_xconst_avx512_nofma_div_vector,
    i64_xconst_avx2_nofma_div_vector,
    i64_xconst_neon_nofma_div_vector,
    i64_xconst_fallback_nofma_div_vector,
    i64_xany_avx512_nofma_div_vector,
    i64_xany_avx2_nofma_div_vector,
    i64_xany_neon_nofma_div_vector,
    i64_xany_fallback_nofma_div_vector,
);

#[cfg(test)]
/// Tests the exposed safe API.
///
/// These are more sanity checks than anything else, because they can change depending
/// on the system running them (as they are runtime selected)
mod tests {
    use std::iter::zip;

    use super::*;
    use crate::math::{AutoMath, Math};

    const DIMS: usize = 77;

    macro_rules! test_arithmetic {
        ($t:ident) => {
            paste::paste! {
                #[test]
                fn [< test_ $t _add_ops >]() {
                    let (l1, l2) = crate::test_utils::get_sample_vectors::<$t>(DIMS);

                    let mut r = vec![$t::default(); DIMS];
                    [<$t _xany_add_value >](1 as $t, &l1, &mut r);
                    assert_eq!(r, l1.iter().map(|v| *v + 1 as $t).collect::<Vec<_>>(), "Add value op miss-match");

                    let mut r = vec![$t::default(); DIMS];
                    [<$t _xany_add_vector >](&l1, &l2, &mut r);
                    let expected = zip(l1, l2).map(|(a, b)| AutoMath::add(a, b)).collect::<Vec<_>>();
                    assert_eq!(r, expected, "Add vctor op miss-match");
                }

                #[test]
                fn [< test_ $t _sub_ops >]() {
                    let (l1, l2) = crate::test_utils::get_sample_vectors::<$t>(DIMS);

                    let mut r = vec![$t::default(); DIMS];
                    [<$t _xany_sub_value >](1 as $t, &l1, &mut r);
                    assert_eq!(r, l1.iter().map(|v| *v - 1 as $t).collect::<Vec<_>>(), "Sub value op miss-match");

                    let mut r = vec![$t::default(); DIMS];
                    [<$t _xany_sub_vector >](&l1, &l2, &mut r);
                    let expected = zip(l1, l2).map(|(a, b)| AutoMath::sub(a, b)).collect::<Vec<_>>();
                    assert_eq!(r, expected, "Sub vector op miss-match");
                }

                #[test]
                fn [< test_ $t _mul_ops >]() {
                    let (l1, l2) = crate::test_utils::get_sample_vectors::<$t>(DIMS);

                    let mut r = vec![$t::default(); DIMS];
                    [<$t _xany_mul_value >](1 as $t, &l1, &mut r);
                    assert_eq!(r, l1.iter().map(|v| *v * 1 as $t).collect::<Vec<_>>(), "Mul value op miss-match");

                    let mut r = vec![$t::default(); DIMS];
                    [<$t _xany_mul_vector >](&l1, &l2, &mut r);
                    let expected = zip(l1, l2).map(|(a, b)| AutoMath::mul(a, b)).collect::<Vec<_>>();
                    assert_eq!(r, expected, "Mul vector op miss-match");
                }

                #[test]
                fn [< test_ $t _div_ops >]() {
                    let (l1, l2) = crate::test_utils::get_sample_vectors::<$t>(DIMS);

                    let mut r = vec![$t::default(); DIMS];
                    [<$t _xany_div_value >](1 as $t, &l1, &mut r);
                    assert_eq!(r, l1.iter().map(|v| *v / 1 as $t).collect::<Vec<_>>(), "Div value op miss-match");

                    let mut r = vec![$t::default(); DIMS];
                    [<$t _xany_div_vector >](&l1, &l2, &mut r);
                    let expected = zip(l1, l2).map(|(a, b)| AutoMath::div(a, b)).collect::<Vec<_>>();
                    assert_eq!(r, expected, "Div vector op miss-match");
                }

            }
        };
    }

    test_arithmetic!(f32);
    test_arithmetic!(f64);
    test_arithmetic!(u8);
    test_arithmetic!(u16);
    test_arithmetic!(u32);
    test_arithmetic!(u64);
    test_arithmetic!(i8);
    test_arithmetic!(i16);
    test_arithmetic!(i32);
    test_arithmetic!(i64);
}
