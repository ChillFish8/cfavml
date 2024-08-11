//! Vector based min/max/sum/norm operations
//!
//! These exported methods are safe to call and select the fastest available instruction set
//! to use at runtime.
//!
//! Both `xconst` and `xany` variants of each operation are provided.
//!
//! The following arithmetic operations are provided:
//!
//! - The squared L2 norm of the vector
//!
//! - Sum vector horizontally
//! - Min vector horizontally
//! - Max vector horizontally
//!
//! - Min vector vertically
//! - Max vector vertically
//!
use crate::buffer::WriteOnlyBuffer;
use crate::danger::*;

macro_rules! export_safe_horizontal_op {
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
        pub fn $const_name<const DIMS: usize>(a: &[$t]) -> $t {
            assert_eq!(a.len(), DIMS, "Provided vector does not match size DIMS");

            unsafe {
                crate::dispatch!(
                    avx512 = $avx512_const_name::<DIMS>,
                    avx2 = $avx2_const_name::<DIMS>,
                    neon = $neon_const_name::<DIMS>,
                    fallback = $fallback_const_name::<DIMS>,
                    args = (a)
                )
            }
        }

        #[doc = concat!("`", stringify!($t), "` ", $desc)]
        pub fn $any_name(a: &[$t]) -> $t {
            unsafe {
                crate::dispatch!(
                    avx512 = $avx512_any_name,
                    avx2 = $avx2_any_name,
                    neon = $neon_any_name,
                    fallback = $fallback_any_name,
                    args = (a)
                )
            }
        }
    };
}

macro_rules! export_safe_vertical_op {
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
            assert_eq!(a.len(), DIMS, "Input vector a does not match size DIMS");
            assert_eq!(b.len(), DIMS, "Input vector b does not match size DIMS");
            assert_eq!(
                result.raw_buffer_len(),
                DIMS,
                "Result vector does not match size DIMS"
            );

            unsafe {
                crate::dispatch!(
                    avx512 = $avx512_const_name::<DIMS>,
                    avx2 = $avx2_const_name::<DIMS>,
                    neon = $neon_const_name::<DIMS>,
                    fallback = $fallback_const_name::<DIMS>,
                    args = (a, b, result)
                )
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
                )
            }
        }
    };
}

macro_rules! export_safe_value_op {
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
            assert_eq!(a.len(), DIMS, "Input vector a does not match size DIMS");
            assert_eq!(
                result.raw_buffer_len(),
                DIMS,
                "Result vector does not match size DIMS"
            );

            unsafe {
                crate::dispatch!(
                    avx512 = $avx512_const_name::<DIMS>,
                    avx2 = $avx2_const_name::<DIMS>,
                    neon = $neon_const_name::<DIMS>,
                    fallback = $fallback_const_name::<DIMS>,
                    args = (value, a, result)
                )
            }
        }

        #[doc = concat!("`", stringify!($t), "` ", $desc)]
        pub fn $any_name(value: $t, a: &[$t], result: impl WriteOnlyBuffer<Item = $t>) {
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
                    args = (value, a, result)
                )
            }
        }
    };
}

export_safe_horizontal_op!(
    description = "Performs a horizontal sum of all elements in vector `a`",
    ty = f32,
    const_name = f32_xconst_sum,
    any_name = f32_xany_sum,
    f32_xconst_avx512_nofma_sum,
    f32_xconst_avx2_nofma_sum,
    f32_xconst_neon_nofma_sum,
    f32_xconst_fallback_nofma_sum,
    f32_xany_avx512_nofma_sum,
    f32_xany_avx2_nofma_sum,
    f32_xany_neon_nofma_sum,
    f32_xany_fallback_nofma_sum,
);
export_safe_horizontal_op!(
    description = "Performs a horizontal max of all elements in vector `a`",
    ty = f32,
    const_name = f32_xconst_max_horizontal,
    any_name = f32_xany_max_horizontal,
    f32_xconst_avx512_nofma_max_horizontal,
    f32_xconst_avx2_nofma_max_horizontal,
    f32_xconst_neon_nofma_max_horizontal,
    f32_xconst_fallback_nofma_max_horizontal,
    f32_xany_avx512_nofma_max_horizontal,
    f32_xany_avx2_nofma_max_horizontal,
    f32_xany_neon_nofma_max_horizontal,
    f32_xany_fallback_nofma_max_horizontal,
);
export_safe_horizontal_op!(
    description = "Performs a horizontal min of all elements in vector `a`",
    ty = f32,
    const_name = f32_xconst_min_horizontal,
    any_name = f32_xany_min_horizontal,
    f32_xconst_avx512_nofma_min_horizontal,
    f32_xconst_avx2_nofma_min_horizontal,
    f32_xconst_neon_nofma_min_horizontal,
    f32_xconst_fallback_nofma_min_horizontal,
    f32_xany_avx512_nofma_min_horizontal,
    f32_xany_avx2_nofma_min_horizontal,
    f32_xany_neon_nofma_min_horizontal,
    f32_xany_fallback_nofma_min_horizontal,
);

export_safe_horizontal_op!(
    description = "Performs a horizontal sum of all elements in vector `a`",
    ty = f64,
    const_name = f64_xconst_sum,
    any_name = f64_xany_sum,
    f64_xconst_avx512_nofma_sum,
    f64_xconst_avx2_nofma_sum,
    f64_xconst_neon_nofma_sum,
    f64_xconst_fallback_nofma_sum,
    f64_xany_avx512_nofma_sum,
    f64_xany_avx2_nofma_sum,
    f64_xany_neon_nofma_sum,
    f64_xany_fallback_nofma_sum,
);
export_safe_horizontal_op!(
    description = "Performs a horizontal max of all elements in vector `a`",
    ty = f64,
    const_name = f64_xconst_max_horizontal,
    any_name = f64_xany_max_horizontal,
    f64_xconst_avx512_nofma_max_horizontal,
    f64_xconst_avx2_nofma_max_horizontal,
    f64_xconst_neon_nofma_max_horizontal,
    f64_xconst_fallback_nofma_max_horizontal,
    f64_xany_avx512_nofma_max_horizontal,
    f64_xany_avx2_nofma_max_horizontal,
    f64_xany_neon_nofma_max_horizontal,
    f64_xany_fallback_nofma_max_horizontal,
);
export_safe_horizontal_op!(
    description = "Performs a horizontal min of all elements in vector `a`",
    ty = f64,
    const_name = f64_xconst_min_horizontal,
    any_name = f64_xany_min_horizontal,
    f64_xconst_avx512_nofma_min_horizontal,
    f64_xconst_avx2_nofma_min_horizontal,
    f64_xconst_neon_nofma_min_horizontal,
    f64_xconst_fallback_nofma_min_horizontal,
    f64_xany_avx512_nofma_min_horizontal,
    f64_xany_avx2_nofma_min_horizontal,
    f64_xany_neon_nofma_min_horizontal,
    f64_xany_fallback_nofma_min_horizontal,
);

export_safe_horizontal_op!(
    description = "Performs a horizontal sum of all elements in vector `a`",
    ty = u8,
    const_name = u8_xconst_sum,
    any_name = u8_xany_sum,
    u8_xconst_avx512_nofma_sum,
    u8_xconst_avx2_nofma_sum,
    u8_xconst_neon_nofma_sum,
    u8_xconst_fallback_nofma_sum,
    u8_xany_avx512_nofma_sum,
    u8_xany_avx2_nofma_sum,
    u8_xany_neon_nofma_sum,
    u8_xany_fallback_nofma_sum,
);
export_safe_horizontal_op!(
    description = "Performs a horizontal max of all elements in vector `a`",
    ty = u8,
    const_name = u8_xconst_max_horizontal,
    any_name = u8_xany_max_horizontal,
    u8_xconst_avx512_nofma_max_horizontal,
    u8_xconst_avx2_nofma_max_horizontal,
    u8_xconst_neon_nofma_max_horizontal,
    u8_xconst_fallback_nofma_max_horizontal,
    u8_xany_avx512_nofma_max_horizontal,
    u8_xany_avx2_nofma_max_horizontal,
    u8_xany_neon_nofma_max_horizontal,
    u8_xany_fallback_nofma_max_horizontal,
);
export_safe_horizontal_op!(
    description = "Performs a horizontal min of all elements in vector `a`",
    ty = u8,
    const_name = u8_xconst_min_horizontal,
    any_name = u8_xany_min_horizontal,
    u8_xconst_avx512_nofma_min_horizontal,
    u8_xconst_avx2_nofma_min_horizontal,
    u8_xconst_neon_nofma_min_horizontal,
    u8_xconst_fallback_nofma_min_horizontal,
    u8_xany_avx512_nofma_min_horizontal,
    u8_xany_avx2_nofma_min_horizontal,
    u8_xany_neon_nofma_min_horizontal,
    u8_xany_fallback_nofma_min_horizontal,
);

export_safe_horizontal_op!(
    description = "Performs a horizontal sum of all elements in vector `a`",
    ty = u16,
    const_name = u16_xconst_sum,
    any_name = u16_xany_sum,
    u16_xconst_avx512_nofma_sum,
    u16_xconst_avx2_nofma_sum,
    u16_xconst_neon_nofma_sum,
    u16_xconst_fallback_nofma_sum,
    u16_xany_avx512_nofma_sum,
    u16_xany_avx2_nofma_sum,
    u16_xany_neon_nofma_sum,
    u16_xany_fallback_nofma_sum,
);
export_safe_horizontal_op!(
    description = "Performs a horizontal max of all elements in vector `a`",
    ty = u16,
    const_name = u16_xconst_max_horizontal,
    any_name = u16_xany_max_horizontal,
    u16_xconst_avx512_nofma_max_horizontal,
    u16_xconst_avx2_nofma_max_horizontal,
    u16_xconst_neon_nofma_max_horizontal,
    u16_xconst_fallback_nofma_max_horizontal,
    u16_xany_avx512_nofma_max_horizontal,
    u16_xany_avx2_nofma_max_horizontal,
    u16_xany_neon_nofma_max_horizontal,
    u16_xany_fallback_nofma_max_horizontal,
);
export_safe_horizontal_op!(
    description = "Performs a horizontal min of all elements in vector `a`",
    ty = u16,
    const_name = u16_xconst_min_horizontal,
    any_name = u16_xany_min_horizontal,
    u16_xconst_avx512_nofma_min_horizontal,
    u16_xconst_avx2_nofma_min_horizontal,
    u16_xconst_neon_nofma_min_horizontal,
    u16_xconst_fallback_nofma_min_horizontal,
    u16_xany_avx512_nofma_min_horizontal,
    u16_xany_avx2_nofma_min_horizontal,
    u16_xany_neon_nofma_min_horizontal,
    u16_xany_fallback_nofma_min_horizontal,
);

export_safe_horizontal_op!(
    description = "Performs a horizontal sum of all elements in vector `a`",
    ty = u32,
    const_name = u32_xconst_sum,
    any_name = u32_xany_sum,
    u32_xconst_avx512_nofma_sum,
    u32_xconst_avx2_nofma_sum,
    u32_xconst_neon_nofma_sum,
    u32_xconst_fallback_nofma_sum,
    u32_xany_avx512_nofma_sum,
    u32_xany_avx2_nofma_sum,
    u32_xany_neon_nofma_sum,
    u32_xany_fallback_nofma_sum,
);
export_safe_horizontal_op!(
    description = "Performs a horizontal max of all elements in vector `a`",
    ty = u32,
    const_name = u32_xconst_max_horizontal,
    any_name = u32_xany_max_horizontal,
    u32_xconst_avx512_nofma_max_horizontal,
    u32_xconst_avx2_nofma_max_horizontal,
    u32_xconst_neon_nofma_max_horizontal,
    u32_xconst_fallback_nofma_max_horizontal,
    u32_xany_avx512_nofma_max_horizontal,
    u32_xany_avx2_nofma_max_horizontal,
    u32_xany_neon_nofma_max_horizontal,
    u32_xany_fallback_nofma_max_horizontal,
);
export_safe_horizontal_op!(
    description = "Performs a horizontal min of all elements in vector `a`",
    ty = u32,
    const_name = u32_xconst_min_horizontal,
    any_name = u32_xany_min_horizontal,
    u32_xconst_avx512_nofma_min_horizontal,
    u32_xconst_avx2_nofma_min_horizontal,
    u32_xconst_neon_nofma_min_horizontal,
    u32_xconst_fallback_nofma_min_horizontal,
    u32_xany_avx512_nofma_min_horizontal,
    u32_xany_avx2_nofma_min_horizontal,
    u32_xany_neon_nofma_min_horizontal,
    u32_xany_fallback_nofma_min_horizontal,
);

export_safe_horizontal_op!(
    description = "Performs a horizontal sum of all elements in vector `a`",
    ty = u64,
    const_name = u64_xconst_sum,
    any_name = u64_xany_sum,
    u64_xconst_avx512_nofma_sum,
    u64_xconst_avx2_nofma_sum,
    u64_xconst_neon_nofma_sum,
    u64_xconst_fallback_nofma_sum,
    u64_xany_avx512_nofma_sum,
    u64_xany_avx2_nofma_sum,
    u64_xany_neon_nofma_sum,
    u64_xany_fallback_nofma_sum,
);
export_safe_horizontal_op!(
    description = "Performs a horizontal max of all elements in vector `a`",
    ty = u64,
    const_name = u64_xconst_max_horizontal,
    any_name = u64_xany_max_horizontal,
    u64_xconst_avx512_nofma_max_horizontal,
    u64_xconst_avx2_nofma_max_horizontal,
    u64_xconst_neon_nofma_max_horizontal,
    u64_xconst_fallback_nofma_max_horizontal,
    u64_xany_avx512_nofma_max_horizontal,
    u64_xany_avx2_nofma_max_horizontal,
    u64_xany_neon_nofma_max_horizontal,
    u64_xany_fallback_nofma_max_horizontal,
);
export_safe_horizontal_op!(
    description = "Performs a horizontal min of all elements in vector `a`",
    ty = u64,
    const_name = u64_xconst_min_horizontal,
    any_name = u64_xany_min_horizontal,
    u64_xconst_avx512_nofma_min_horizontal,
    u64_xconst_avx2_nofma_min_horizontal,
    u64_xconst_neon_nofma_min_horizontal,
    u64_xconst_fallback_nofma_min_horizontal,
    u64_xany_avx512_nofma_min_horizontal,
    u64_xany_avx2_nofma_min_horizontal,
    u64_xany_neon_nofma_min_horizontal,
    u64_xany_fallback_nofma_min_horizontal,
);

export_safe_horizontal_op!(
    description = "Performs a horizontal sum of all elements in vector `a`",
    ty = i8,
    const_name = i8_xconst_sum,
    any_name = i8_xany_sum,
    i8_xconst_avx512_nofma_sum,
    i8_xconst_avx2_nofma_sum,
    i8_xconst_neon_nofma_sum,
    i8_xconst_fallback_nofma_sum,
    i8_xany_avx512_nofma_sum,
    i8_xany_avx2_nofma_sum,
    i8_xany_neon_nofma_sum,
    i8_xany_fallback_nofma_sum,
);
export_safe_horizontal_op!(
    description = "Performs a horizontal max of all elements in vector `a`",
    ty = i8,
    const_name = i8_xconst_max_horizontal,
    any_name = i8_xany_max_horizontal,
    i8_xconst_avx512_nofma_max_horizontal,
    i8_xconst_avx2_nofma_max_horizontal,
    i8_xconst_neon_nofma_max_horizontal,
    i8_xconst_fallback_nofma_max_horizontal,
    i8_xany_avx512_nofma_max_horizontal,
    i8_xany_avx2_nofma_max_horizontal,
    i8_xany_neon_nofma_max_horizontal,
    i8_xany_fallback_nofma_max_horizontal,
);
export_safe_horizontal_op!(
    description = "Performs a horizontal min of all elements in vector `a`",
    ty = i8,
    const_name = i8_xconst_min_horizontal,
    any_name = i8_xany_min_horizontal,
    i8_xconst_avx512_nofma_min_horizontal,
    i8_xconst_avx2_nofma_min_horizontal,
    i8_xconst_neon_nofma_min_horizontal,
    i8_xconst_fallback_nofma_min_horizontal,
    i8_xany_avx512_nofma_min_horizontal,
    i8_xany_avx2_nofma_min_horizontal,
    i8_xany_neon_nofma_min_horizontal,
    i8_xany_fallback_nofma_min_horizontal,
);

export_safe_horizontal_op!(
    description = "Performs a horizontal sum of all elements in vector `a`",
    ty = i16,
    const_name = i16_xconst_sum,
    any_name = i16_xany_sum,
    i16_xconst_avx512_nofma_sum,
    i16_xconst_avx2_nofma_sum,
    i16_xconst_neon_nofma_sum,
    i16_xconst_fallback_nofma_sum,
    i16_xany_avx512_nofma_sum,
    i16_xany_avx2_nofma_sum,
    i16_xany_neon_nofma_sum,
    i16_xany_fallback_nofma_sum,
);
export_safe_horizontal_op!(
    description = "Performs a horizontal max of all elements in vector `a`",
    ty = i16,
    const_name = i16_xconst_max_horizontal,
    any_name = i16_xany_max_horizontal,
    i16_xconst_avx512_nofma_max_horizontal,
    i16_xconst_avx2_nofma_max_horizontal,
    i16_xconst_neon_nofma_max_horizontal,
    i16_xconst_fallback_nofma_max_horizontal,
    i16_xany_avx512_nofma_max_horizontal,
    i16_xany_avx2_nofma_max_horizontal,
    i16_xany_neon_nofma_max_horizontal,
    i16_xany_fallback_nofma_max_horizontal,
);
export_safe_horizontal_op!(
    description = "Performs a horizontal min of all elements in vector `a`",
    ty = i16,
    const_name = i16_xconst_min_horizontal,
    any_name = i16_xany_min_horizontal,
    i16_xconst_avx512_nofma_min_horizontal,
    i16_xconst_avx2_nofma_min_horizontal,
    i16_xconst_neon_nofma_min_horizontal,
    i16_xconst_fallback_nofma_min_horizontal,
    i16_xany_avx512_nofma_min_horizontal,
    i16_xany_avx2_nofma_min_horizontal,
    i16_xany_neon_nofma_min_horizontal,
    i16_xany_fallback_nofma_min_horizontal,
);

export_safe_horizontal_op!(
    description = "Performs a horizontal sum of all elements in vector `a`",
    ty = i32,
    const_name = i32_xconst_sum,
    any_name = i32_xany_sum,
    i32_xconst_avx512_nofma_sum,
    i32_xconst_avx2_nofma_sum,
    i32_xconst_neon_nofma_sum,
    i32_xconst_fallback_nofma_sum,
    i32_xany_avx512_nofma_sum,
    i32_xany_avx2_nofma_sum,
    i32_xany_neon_nofma_sum,
    i32_xany_fallback_nofma_sum,
);
export_safe_horizontal_op!(
    description = "Performs a horizontal max of all elements in vector `a`",
    ty = i32,
    const_name = i32_xconst_max_horizontal,
    any_name = i32_xany_max_horizontal,
    i32_xconst_avx512_nofma_max_horizontal,
    i32_xconst_avx2_nofma_max_horizontal,
    i32_xconst_neon_nofma_max_horizontal,
    i32_xconst_fallback_nofma_max_horizontal,
    i32_xany_avx512_nofma_max_horizontal,
    i32_xany_avx2_nofma_max_horizontal,
    i32_xany_neon_nofma_max_horizontal,
    i32_xany_fallback_nofma_max_horizontal,
);
export_safe_horizontal_op!(
    description = "Performs a horizontal min of all elements in vector `a`",
    ty = i32,
    const_name = i32_xconst_min_horizontal,
    any_name = i32_xany_min_horizontal,
    i32_xconst_avx512_nofma_min_horizontal,
    i32_xconst_avx2_nofma_min_horizontal,
    i32_xconst_neon_nofma_min_horizontal,
    i32_xconst_fallback_nofma_min_horizontal,
    i32_xany_avx512_nofma_min_horizontal,
    i32_xany_avx2_nofma_min_horizontal,
    i32_xany_neon_nofma_min_horizontal,
    i32_xany_fallback_nofma_min_horizontal,
);

export_safe_horizontal_op!(
    description = "Performs a horizontal sum of all elements in vector `a`",
    ty = i64,
    const_name = i64_xconst_sum,
    any_name = i64_xany_sum,
    i64_xconst_avx512_nofma_sum,
    i64_xconst_avx2_nofma_sum,
    i64_xconst_neon_nofma_sum,
    i64_xconst_fallback_nofma_sum,
    i64_xany_avx512_nofma_sum,
    i64_xany_avx2_nofma_sum,
    i64_xany_neon_nofma_sum,
    i64_xany_fallback_nofma_sum,
);
export_safe_horizontal_op!(
    description = "Performs a horizontal max of all elements in vector `a`",
    ty = i64,
    const_name = i64_xconst_max_horizontal,
    any_name = i64_xany_max_horizontal,
    i64_xconst_avx512_nofma_max_horizontal,
    i64_xconst_avx2_nofma_max_horizontal,
    i64_xconst_neon_nofma_max_horizontal,
    i64_xconst_fallback_nofma_max_horizontal,
    i64_xany_avx512_nofma_max_horizontal,
    i64_xany_avx2_nofma_max_horizontal,
    i64_xany_neon_nofma_max_horizontal,
    i64_xany_fallback_nofma_max_horizontal,
);
export_safe_horizontal_op!(
    description = "Performs a horizontal min of all elements in vector `a`",
    ty = i64,
    const_name = i64_xconst_min_horizontal,
    any_name = i64_xany_min_horizontal,
    i64_xconst_avx512_nofma_min_horizontal,
    i64_xconst_avx2_nofma_min_horizontal,
    i64_xconst_neon_nofma_min_horizontal,
    i64_xconst_fallback_nofma_min_horizontal,
    i64_xany_avx512_nofma_min_horizontal,
    i64_xany_avx2_nofma_min_horizontal,
    i64_xany_neon_nofma_min_horizontal,
    i64_xany_fallback_nofma_min_horizontal,
);

export_safe_vertical_op!(
    description =
        "Performs a vertical max of each respective element of vectors `a` and `b`",
    ty = f32,
    const_name = f32_xconst_max_vertical,
    any_name = f32_xany_max_vertical,
    f32_xconst_avx512_nofma_max_vertical,
    f32_xconst_avx2_nofma_max_vertical,
    f32_xconst_neon_nofma_max_vertical,
    f32_xconst_fallback_nofma_max_vertical,
    f32_xany_avx512_nofma_max_vertical,
    f32_xany_avx2_nofma_max_vertical,
    f32_xany_neon_nofma_max_vertical,
    f32_xany_fallback_nofma_max_vertical,
);
export_safe_vertical_op!(
    description =
        "Performs a vertical min of each respective element in vectors `a` and `b`",
    ty = f32,
    const_name = f32_xconst_min_vertical,
    any_name = f32_xany_min_vertical,
    f32_xconst_avx512_nofma_min_vertical,
    f32_xconst_avx2_nofma_min_vertical,
    f32_xconst_neon_nofma_min_vertical,
    f32_xconst_fallback_nofma_min_vertical,
    f32_xany_avx512_nofma_min_vertical,
    f32_xany_avx2_nofma_min_vertical,
    f32_xany_neon_nofma_min_vertical,
    f32_xany_fallback_nofma_min_vertical,
);

export_safe_vertical_op!(
    description =
        "Performs a vertical max of each respective element of vectors `a` and `b`",
    ty = f64,
    const_name = f64_xconst_max_vertical,
    any_name = f64_xany_max_vertical,
    f64_xconst_avx512_nofma_max_vertical,
    f64_xconst_avx2_nofma_max_vertical,
    f64_xconst_neon_nofma_max_vertical,
    f64_xconst_fallback_nofma_max_vertical,
    f64_xany_avx512_nofma_max_vertical,
    f64_xany_avx2_nofma_max_vertical,
    f64_xany_neon_nofma_max_vertical,
    f64_xany_fallback_nofma_max_vertical,
);
export_safe_vertical_op!(
    description =
        "Performs a vertical min of each respective element in vectors `a` and `b`",
    ty = f64,
    const_name = f64_xconst_min_vertical,
    any_name = f64_xany_min_vertical,
    f64_xconst_avx512_nofma_min_vertical,
    f64_xconst_avx2_nofma_min_vertical,
    f64_xconst_neon_nofma_min_vertical,
    f64_xconst_fallback_nofma_min_vertical,
    f64_xany_avx512_nofma_min_vertical,
    f64_xany_avx2_nofma_min_vertical,
    f64_xany_neon_nofma_min_vertical,
    f64_xany_fallback_nofma_min_vertical,
);

export_safe_vertical_op!(
    description =
        "Performs a vertical max of each respective element of vectors `a` and `b`",
    ty = u8,
    const_name = u8_xconst_max_vertical,
    any_name = u8_xany_max_vertical,
    u8_xconst_avx512_nofma_max_vertical,
    u8_xconst_avx2_nofma_max_vertical,
    u8_xconst_neon_nofma_max_vertical,
    u8_xconst_fallback_nofma_max_vertical,
    u8_xany_avx512_nofma_max_vertical,
    u8_xany_avx2_nofma_max_vertical,
    u8_xany_neon_nofma_max_vertical,
    u8_xany_fallback_nofma_max_vertical,
);
export_safe_vertical_op!(
    description =
        "Performs a vertical min of each respective element in vectors `a` and `b`",
    ty = u8,
    const_name = u8_xconst_min_vertical,
    any_name = u8_xany_min_vertical,
    u8_xconst_avx512_nofma_min_vertical,
    u8_xconst_avx2_nofma_min_vertical,
    u8_xconst_neon_nofma_min_vertical,
    u8_xconst_fallback_nofma_min_vertical,
    u8_xany_avx512_nofma_min_vertical,
    u8_xany_avx2_nofma_min_vertical,
    u8_xany_neon_nofma_min_vertical,
    u8_xany_fallback_nofma_min_vertical,
);

export_safe_vertical_op!(
    description =
        "Performs a vertical max of each respective element of vectors `a` and `b`",
    ty = u16,
    const_name = u16_xconst_max_vertical,
    any_name = u16_xany_max_vertical,
    u16_xconst_avx512_nofma_max_vertical,
    u16_xconst_avx2_nofma_max_vertical,
    u16_xconst_neon_nofma_max_vertical,
    u16_xconst_fallback_nofma_max_vertical,
    u16_xany_avx512_nofma_max_vertical,
    u16_xany_avx2_nofma_max_vertical,
    u16_xany_neon_nofma_max_vertical,
    u16_xany_fallback_nofma_max_vertical,
);
export_safe_vertical_op!(
    description =
        "Performs a vertical min of each respective element in vectors `a` and `b`",
    ty = u16,
    const_name = u16_xconst_min_vertical,
    any_name = u16_xany_min_vertical,
    u16_xconst_avx512_nofma_min_vertical,
    u16_xconst_avx2_nofma_min_vertical,
    u16_xconst_neon_nofma_min_vertical,
    u16_xconst_fallback_nofma_min_vertical,
    u16_xany_avx512_nofma_min_vertical,
    u16_xany_avx2_nofma_min_vertical,
    u16_xany_neon_nofma_min_vertical,
    u16_xany_fallback_nofma_min_vertical,
);

export_safe_vertical_op!(
    description =
        "Performs a vertical max of each respective element of vectors `a` and `b`",
    ty = u32,
    const_name = u32_xconst_max_vertical,
    any_name = u32_xany_max_vertical,
    u32_xconst_avx512_nofma_max_vertical,
    u32_xconst_avx2_nofma_max_vertical,
    u32_xconst_neon_nofma_max_vertical,
    u32_xconst_fallback_nofma_max_vertical,
    u32_xany_avx512_nofma_max_vertical,
    u32_xany_avx2_nofma_max_vertical,
    u32_xany_neon_nofma_max_vertical,
    u32_xany_fallback_nofma_max_vertical,
);
export_safe_vertical_op!(
    description =
        "Performs a vertical min of each respective element in vectors `a` and `b`",
    ty = u32,
    const_name = u32_xconst_min_vertical,
    any_name = u32_xany_min_vertical,
    u32_xconst_avx512_nofma_min_vertical,
    u32_xconst_avx2_nofma_min_vertical,
    u32_xconst_neon_nofma_min_vertical,
    u32_xconst_fallback_nofma_min_vertical,
    u32_xany_avx512_nofma_min_vertical,
    u32_xany_avx2_nofma_min_vertical,
    u32_xany_neon_nofma_min_vertical,
    u32_xany_fallback_nofma_min_vertical,
);

export_safe_vertical_op!(
    description =
        "Performs a vertical max of each respective element of vectors `a` and `b`",
    ty = u64,
    const_name = u64_xconst_max_vertical,
    any_name = u64_xany_max_vertical,
    u64_xconst_avx512_nofma_max_vertical,
    u64_xconst_avx2_nofma_max_vertical,
    u64_xconst_neon_nofma_max_vertical,
    u64_xconst_fallback_nofma_max_vertical,
    u64_xany_avx512_nofma_max_vertical,
    u64_xany_avx2_nofma_max_vertical,
    u64_xany_neon_nofma_max_vertical,
    u64_xany_fallback_nofma_max_vertical,
);
export_safe_vertical_op!(
    description =
        "Performs a vertical min of each respective element in vectors `a` and `b`",
    ty = u64,
    const_name = u64_xconst_min_vertical,
    any_name = u64_xany_min_vertical,
    u64_xconst_avx512_nofma_min_vertical,
    u64_xconst_avx2_nofma_min_vertical,
    u64_xconst_neon_nofma_min_vertical,
    u64_xconst_fallback_nofma_min_vertical,
    u64_xany_avx512_nofma_min_vertical,
    u64_xany_avx2_nofma_min_vertical,
    u64_xany_neon_nofma_min_vertical,
    u64_xany_fallback_nofma_min_vertical,
);

export_safe_vertical_op!(
    description =
        "Performs a vertical max of each respective element of vectors `a` and `b`",
    ty = i8,
    const_name = i8_xconst_max_vertical,
    any_name = i8_xany_max_vertical,
    i8_xconst_avx512_nofma_max_vertical,
    i8_xconst_avx2_nofma_max_vertical,
    i8_xconst_neon_nofma_max_vertical,
    i8_xconst_fallback_nofma_max_vertical,
    i8_xany_avx512_nofma_max_vertical,
    i8_xany_avx2_nofma_max_vertical,
    i8_xany_neon_nofma_max_vertical,
    i8_xany_fallback_nofma_max_vertical,
);
export_safe_vertical_op!(
    description =
        "Performs a vertical min of each respective element in vectors `a` and `b`",
    ty = i8,
    const_name = i8_xconst_min_vertical,
    any_name = i8_xany_min_vertical,
    i8_xconst_avx512_nofma_min_vertical,
    i8_xconst_avx2_nofma_min_vertical,
    i8_xconst_neon_nofma_min_vertical,
    i8_xconst_fallback_nofma_min_vertical,
    i8_xany_avx512_nofma_min_vertical,
    i8_xany_avx2_nofma_min_vertical,
    i8_xany_neon_nofma_min_vertical,
    i8_xany_fallback_nofma_min_vertical,
);

export_safe_vertical_op!(
    description =
        "Performs a vertical max of each respective element of vectors `a` and `b`",
    ty = i16,
    const_name = i16_xconst_max_vertical,
    any_name = i16_xany_max_vertical,
    i16_xconst_avx512_nofma_max_vertical,
    i16_xconst_avx2_nofma_max_vertical,
    i16_xconst_neon_nofma_max_vertical,
    i16_xconst_fallback_nofma_max_vertical,
    i16_xany_avx512_nofma_max_vertical,
    i16_xany_avx2_nofma_max_vertical,
    i16_xany_neon_nofma_max_vertical,
    i16_xany_fallback_nofma_max_vertical,
);
export_safe_vertical_op!(
    description =
        "Performs a vertical min of each respective element in vectors `a` and `b`",
    ty = i16,
    const_name = i16_xconst_min_vertical,
    any_name = i16_xany_min_vertical,
    i16_xconst_avx512_nofma_min_vertical,
    i16_xconst_avx2_nofma_min_vertical,
    i16_xconst_neon_nofma_min_vertical,
    i16_xconst_fallback_nofma_min_vertical,
    i16_xany_avx512_nofma_min_vertical,
    i16_xany_avx2_nofma_min_vertical,
    i16_xany_neon_nofma_min_vertical,
    i16_xany_fallback_nofma_min_vertical,
);

export_safe_vertical_op!(
    description =
        "Performs a vertical max of each respective element of vectors `a` and `b`",
    ty = i32,
    const_name = i32_xconst_max_vertical,
    any_name = i32_xany_max_vertical,
    i32_xconst_avx512_nofma_max_vertical,
    i32_xconst_avx2_nofma_max_vertical,
    i32_xconst_neon_nofma_max_vertical,
    i32_xconst_fallback_nofma_max_vertical,
    i32_xany_avx512_nofma_max_vertical,
    i32_xany_avx2_nofma_max_vertical,
    i32_xany_neon_nofma_max_vertical,
    i32_xany_fallback_nofma_max_vertical,
);
export_safe_vertical_op!(
    description =
        "Performs a vertical min of each respective element in vectors `a` and `b`",
    ty = i32,
    const_name = i32_xconst_min_vertical,
    any_name = i32_xany_min_vertical,
    i32_xconst_avx512_nofma_min_vertical,
    i32_xconst_avx2_nofma_min_vertical,
    i32_xconst_neon_nofma_min_vertical,
    i32_xconst_fallback_nofma_min_vertical,
    i32_xany_avx512_nofma_min_vertical,
    i32_xany_avx2_nofma_min_vertical,
    i32_xany_neon_nofma_min_vertical,
    i32_xany_fallback_nofma_min_vertical,
);

export_safe_vertical_op!(
    description =
        "Performs a vertical max of each respective element of vectors `a` and `b`",
    ty = i64,
    const_name = i64_xconst_max_vertical,
    any_name = i64_xany_max_vertical,
    i64_xconst_avx512_nofma_max_vertical,
    i64_xconst_avx2_nofma_max_vertical,
    i64_xconst_neon_nofma_max_vertical,
    i64_xconst_fallback_nofma_max_vertical,
    i64_xany_avx512_nofma_max_vertical,
    i64_xany_avx2_nofma_max_vertical,
    i64_xany_neon_nofma_max_vertical,
    i64_xany_fallback_nofma_max_vertical,
);
export_safe_vertical_op!(
    description =
        "Performs a vertical min of each respective element in vectors `a` and `b`",
    ty = i64,
    const_name = i64_xconst_min_vertical,
    any_name = i64_xany_min_vertical,
    i64_xconst_avx512_nofma_min_vertical,
    i64_xconst_avx2_nofma_min_vertical,
    i64_xconst_neon_nofma_min_vertical,
    i64_xconst_fallback_nofma_min_vertical,
    i64_xany_avx512_nofma_min_vertical,
    i64_xany_avx2_nofma_min_vertical,
    i64_xany_neon_nofma_min_vertical,
    i64_xany_fallback_nofma_min_vertical,
);

export_safe_value_op!(
    description =
        "Performs a vertical min of a provided vector and a single broadcast value",
    ty = f32,
    const_name = f32_xconst_max_value,
    any_name = f32_xany_max_value,
    f32_xconst_avx512_nofma_max_value,
    f32_xconst_avx2_nofma_max_value,
    f32_xconst_neon_nofma_max_value,
    f32_xconst_fallback_nofma_max_value,
    f32_xany_avx512_nofma_max_value,
    f32_xany_avx2_nofma_max_value,
    f32_xany_neon_nofma_max_value,
    f32_xany_fallback_nofma_max_value,
);
export_safe_value_op!(
    description =
        "Performs a vertical min of a provided vector and a single broadcast value",
    ty = f32,
    const_name = f32_xconst_min_value,
    any_name = f32_xany_min_value,
    f32_xconst_avx512_nofma_min_value,
    f32_xconst_avx2_nofma_min_value,
    f32_xconst_neon_nofma_min_value,
    f32_xconst_fallback_nofma_min_value,
    f32_xany_avx512_nofma_min_value,
    f32_xany_avx2_nofma_min_value,
    f32_xany_neon_nofma_min_value,
    f32_xany_fallback_nofma_min_value,
);

export_safe_value_op!(
    description =
        "Performs a vertical max of a provided vector and a single broadcast value",
    ty = f64,
    const_name = f64_xconst_max_value,
    any_name = f64_xany_max_value,
    f64_xconst_avx512_nofma_max_value,
    f64_xconst_avx2_nofma_max_value,
    f64_xconst_neon_nofma_max_value,
    f64_xconst_fallback_nofma_max_value,
    f64_xany_avx512_nofma_max_value,
    f64_xany_avx2_nofma_max_value,
    f64_xany_neon_nofma_max_value,
    f64_xany_fallback_nofma_max_value,
);
export_safe_value_op!(
    description =
        "Performs a vertical min of a provided vector and a single broadcast value",
    ty = f64,
    const_name = f64_xconst_min_value,
    any_name = f64_xany_min_value,
    f64_xconst_avx512_nofma_min_value,
    f64_xconst_avx2_nofma_min_value,
    f64_xconst_neon_nofma_min_value,
    f64_xconst_fallback_nofma_min_value,
    f64_xany_avx512_nofma_min_value,
    f64_xany_avx2_nofma_min_value,
    f64_xany_neon_nofma_min_value,
    f64_xany_fallback_nofma_min_value,
);

export_safe_value_op!(
    description =
        "Performs a vertical min of a provided vector and a single broadcast value",
    ty = i8,
    const_name = i8_xconst_max_value,
    any_name = i8_xany_max_value,
    i8_xconst_avx512_nofma_max_value,
    i8_xconst_avx2_nofma_max_value,
    i8_xconst_neon_nofma_max_value,
    i8_xconst_fallback_nofma_max_value,
    i8_xany_avx512_nofma_max_value,
    i8_xany_avx2_nofma_max_value,
    i8_xany_neon_nofma_max_value,
    i8_xany_fallback_nofma_max_value,
);
export_safe_value_op!(
    description =
        "Performs a vertical min of a provided vector and a single broadcast value",
    ty = i8,
    const_name = i8_xconst_min_value,
    any_name = i8_xany_min_value,
    i8_xconst_avx512_nofma_min_value,
    i8_xconst_avx2_nofma_min_value,
    i8_xconst_neon_nofma_min_value,
    i8_xconst_fallback_nofma_min_value,
    i8_xany_avx512_nofma_min_value,
    i8_xany_avx2_nofma_min_value,
    i8_xany_neon_nofma_min_value,
    i8_xany_fallback_nofma_min_value,
);

export_safe_value_op!(
    description =
        "Performs a vertical max of a provided vector and a single broadcast value",
    ty = i16,
    const_name = i16_xconst_max_value,
    any_name = i16_xany_max_value,
    i16_xconst_avx512_nofma_max_value,
    i16_xconst_avx2_nofma_max_value,
    i16_xconst_neon_nofma_max_value,
    i16_xconst_fallback_nofma_max_value,
    i16_xany_avx512_nofma_max_value,
    i16_xany_avx2_nofma_max_value,
    i16_xany_neon_nofma_max_value,
    i16_xany_fallback_nofma_max_value,
);
export_safe_value_op!(
    description =
        "Performs a vertical min of a provided vector and a single broadcast value",
    ty = i16,
    const_name = i16_xconst_min_value,
    any_name = i16_xany_min_value,
    i16_xconst_avx512_nofma_min_value,
    i16_xconst_avx2_nofma_min_value,
    i16_xconst_neon_nofma_min_value,
    i16_xconst_fallback_nofma_min_value,
    i16_xany_avx512_nofma_min_value,
    i16_xany_avx2_nofma_min_value,
    i16_xany_neon_nofma_min_value,
    i16_xany_fallback_nofma_min_value,
);

export_safe_value_op!(
    description =
        "Performs a vertical max of a provided vector and a single broadcast value",
    ty = i32,
    const_name = i32_xconst_max_value,
    any_name = i32_xany_max_value,
    i32_xconst_avx512_nofma_max_value,
    i32_xconst_avx2_nofma_max_value,
    i32_xconst_neon_nofma_max_value,
    i32_xconst_fallback_nofma_max_value,
    i32_xany_avx512_nofma_max_value,
    i32_xany_avx2_nofma_max_value,
    i32_xany_neon_nofma_max_value,
    i32_xany_fallback_nofma_max_value,
);
export_safe_value_op!(
    description =
        "Performs a vertical min of a provided vector and a single broadcast value",
    ty = i32,
    const_name = i32_xconst_min_value,
    any_name = i32_xany_min_value,
    i32_xconst_avx512_nofma_min_value,
    i32_xconst_avx2_nofma_min_value,
    i32_xconst_neon_nofma_min_value,
    i32_xconst_fallback_nofma_min_value,
    i32_xany_avx512_nofma_min_value,
    i32_xany_avx2_nofma_min_value,
    i32_xany_neon_nofma_min_value,
    i32_xany_fallback_nofma_min_value,
);

export_safe_value_op!(
    description =
        "Performs a vertical max of a provided vector and a single broadcast value",
    ty = i64,
    const_name = i64_xconst_max_value,
    any_name = i64_xany_max_value,
    i64_xconst_avx512_nofma_max_value,
    i64_xconst_avx2_nofma_max_value,
    i64_xconst_neon_nofma_max_value,
    i64_xconst_fallback_nofma_max_value,
    i64_xany_avx512_nofma_max_value,
    i64_xany_avx2_nofma_max_value,
    i64_xany_neon_nofma_max_value,
    i64_xany_fallback_nofma_max_value,
);
export_safe_value_op!(
    description =
        "Performs a vertical min of a provided vector and a single broadcast value",
    ty = i64,
    const_name = i64_xconst_min_value,
    any_name = i64_xany_min_value,
    i64_xconst_avx512_nofma_min_value,
    i64_xconst_avx2_nofma_min_value,
    i64_xconst_neon_nofma_min_value,
    i64_xconst_fallback_nofma_min_value,
    i64_xany_avx512_nofma_min_value,
    i64_xany_avx2_nofma_min_value,
    i64_xany_neon_nofma_min_value,
    i64_xany_fallback_nofma_min_value,
);

export_safe_value_op!(
    description =
        "Performs a vertical max of a provided vector and a single broadcast value",
    ty = u8,
    const_name = u8_xconst_max_value,
    any_name = u8_xany_max_value,
    u8_xconst_avx512_nofma_max_value,
    u8_xconst_avx2_nofma_max_value,
    u8_xconst_neon_nofma_max_value,
    u8_xconst_fallback_nofma_max_value,
    u8_xany_avx512_nofma_max_value,
    u8_xany_avx2_nofma_max_value,
    u8_xany_neon_nofma_max_value,
    u8_xany_fallback_nofma_max_value,
);
export_safe_value_op!(
    description =
        "Performs a vertical min of a provided vector and a single broadcast value",
    ty = u8,
    const_name = u8_xconst_min_value,
    any_name = u8_xany_min_value,
    u8_xconst_avx512_nofma_min_value,
    u8_xconst_avx2_nofma_min_value,
    u8_xconst_neon_nofma_min_value,
    u8_xconst_fallback_nofma_min_value,
    u8_xany_avx512_nofma_min_value,
    u8_xany_avx2_nofma_min_value,
    u8_xany_neon_nofma_min_value,
    u8_xany_fallback_nofma_min_value,
);

export_safe_value_op!(
    description =
        "Performs a vertical max of a provided vector and a single broadcast value",
    ty = u16,
    const_name = u16_xconst_max_value,
    any_name = u16_xany_max_value,
    u16_xconst_avx512_nofma_max_value,
    u16_xconst_avx2_nofma_max_value,
    u16_xconst_neon_nofma_max_value,
    u16_xconst_fallback_nofma_max_value,
    u16_xany_avx512_nofma_max_value,
    u16_xany_avx2_nofma_max_value,
    u16_xany_neon_nofma_max_value,
    u16_xany_fallback_nofma_max_value,
);
export_safe_value_op!(
    description =
        "Performs a vertical min of a provided vector and a single broadcast value",
    ty = u16,
    const_name = u16_xconst_min_value,
    any_name = u16_xany_min_value,
    u16_xconst_avx512_nofma_min_value,
    u16_xconst_avx2_nofma_min_value,
    u16_xconst_neon_nofma_min_value,
    u16_xconst_fallback_nofma_min_value,
    u16_xany_avx512_nofma_min_value,
    u16_xany_avx2_nofma_min_value,
    u16_xany_neon_nofma_min_value,
    u16_xany_fallback_nofma_min_value,
);

export_safe_value_op!(
    description =
        "Performs a vertical max of a provided vector and a single broadcast value",
    ty = u32,
    const_name = u32_xconst_max_value,
    any_name = u32_xany_max_value,
    u32_xconst_avx512_nofma_max_value,
    u32_xconst_avx2_nofma_max_value,
    u32_xconst_neon_nofma_max_value,
    u32_xconst_fallback_nofma_max_value,
    u32_xany_avx512_nofma_max_value,
    u32_xany_avx2_nofma_max_value,
    u32_xany_neon_nofma_max_value,
    u32_xany_fallback_nofma_max_value,
);
export_safe_value_op!(
    description =
        "Performs a vertical min of a provided vector and a single broadcast value",
    ty = u32,
    const_name = u32_xconst_min_value,
    any_name = u32_xany_min_value,
    u32_xconst_avx512_nofma_min_value,
    u32_xconst_avx2_nofma_min_value,
    u32_xconst_neon_nofma_min_value,
    u32_xconst_fallback_nofma_min_value,
    u32_xany_avx512_nofma_min_value,
    u32_xany_avx2_nofma_min_value,
    u32_xany_neon_nofma_min_value,
    u32_xany_fallback_nofma_min_value,
);

export_safe_value_op!(
    description =
        "Performs a vertical max of a provided vector and a single broadcast value",
    ty = u64,
    const_name = u64_xconst_max_value,
    any_name = u64_xany_max_value,
    u64_xconst_avx512_nofma_max_value,
    u64_xconst_avx2_nofma_max_value,
    u64_xconst_neon_nofma_max_value,
    u64_xconst_fallback_nofma_max_value,
    u64_xany_avx512_nofma_max_value,
    u64_xany_avx2_nofma_max_value,
    u64_xany_neon_nofma_max_value,
    u64_xany_fallback_nofma_max_value,
);
export_safe_value_op!(
    description =
        "Performs a vertical min of a provided vector and a single broadcast value",
    ty = u64,
    const_name = u64_xconst_min_value,
    any_name = u64_xany_min_value,
    u64_xconst_avx512_nofma_min_value,
    u64_xconst_avx2_nofma_min_value,
    u64_xconst_neon_nofma_min_value,
    u64_xconst_fallback_nofma_min_value,
    u64_xany_avx512_nofma_min_value,
    u64_xany_avx2_nofma_min_value,
    u64_xany_neon_nofma_min_value,
    u64_xany_fallback_nofma_min_value,
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

    const DIMS: usize = 67;

    macro_rules! test_min_max_sum {
        ($t:ident) => {
            paste::paste! {
                #[test]
                fn [< test_ $t _min_ops >]() {
                    let (l1, l2) = crate::test_utils::get_sample_vectors::<$t>(DIMS);

                    let res = [<$t _xany_min_horizontal >](&l1);
                    assert_eq!(res, l1.iter().fold(AutoMath::max(), |a, b| AutoMath::cmp_min(a, *b)), "Min value op miss-match");

                    let mut r = vec![$t::default(); DIMS];
                    [<$t _xany_min_value >](AutoMath::zero(), &l1, &mut r);
                    let expected = l1.iter()
                        .copied()
                        .map(|a| AutoMath::cmp_min(a, AutoMath::zero()))
                        .collect::<Vec<_>>();
                    assert_eq!(r, expected, "Min vector by value op miss-match");

                    let mut r = vec![$t::default(); DIMS];
                    [<$t _xany_min_vertical >](&l1, &l2, &mut r);
                    let expected = zip(l1, l2).map(|(a, b)| AutoMath::cmp_min(a, b)).collect::<Vec<_>>();
                    assert_eq!(r, expected, "Min vector op miss-match");
                }

                #[test]
                fn [< test_ $t _max_ops >]() {
                    let (l1, l2) = crate::test_utils::get_sample_vectors::<$t>(DIMS);

                    let res = [<$t _xany_max_horizontal >](&l1);
                    assert_eq!(res, l1.iter().fold(AutoMath::min(), |a, b| AutoMath::cmp_max(a, *b)), "Max value op miss-match");

                    let mut r = vec![$t::default(); DIMS];
                    [<$t _xany_max_value >](AutoMath::zero(), &l1, &mut r);
                    let expected = l1.iter()
                        .copied()
                        .map(|a| AutoMath::cmp_max(a, AutoMath::zero()))
                        .collect::<Vec<_>>();
                    assert_eq!(r, expected, "Min vector by value op miss-match");

                    let mut r = vec![$t::default(); DIMS];
                    [<$t _xany_max_vertical >](&l1, &l2, &mut r);
                    let expected = zip(l1, l2).map(|(a, b)| AutoMath::cmp_max(a, b)).collect::<Vec<_>>();
                    assert_eq!(r, expected, "Min vector op miss-match");
                }

                #[test]
                fn [< test_ $t _sum_ops >]() {
                    let (l1, _) = crate::test_utils::get_sample_vectors::<$t>(DIMS);

                    let res = [<$t _xany_sum >](&l1);
                    assert_eq!(res as f32, l1.iter().fold(AutoMath::zero(), |a, b| AutoMath::add(a, *b)) as f32, "Sum horizontal op miss-match");
                }
            }
        };
    }

    test_min_max_sum!(f32);
    test_min_max_sum!(f64);
    test_min_max_sum!(u8);
    test_min_max_sum!(u16);
    test_min_max_sum!(u32);
    test_min_max_sum!(u64);
    test_min_max_sum!(i8);
    test_min_max_sum!(i16);
    test_min_max_sum!(i32);
    test_min_max_sum!(i64);
}
