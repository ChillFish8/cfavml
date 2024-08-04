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
use crate::danger::*;

// I admit the macro setup here is a bit cursed and definitely not DRY.
macro_rules! export_safe_fma_norm_op {
    (
        description = $desc:expr,
        ty = $t:ty,
        const_name = $const_name:ident,
        any_name = $any_name:ident,
        $avx512_const_name:ident,
        $avx2fma_const_name:ident,
        $avx2_const_name:ident,
        $neon_const_name:ident,
        $fallback_const_name:ident,
        $avx512_any_name:ident,
        $avx2fma_any_name:ident,
        $avx2_any_name:ident,
        $neon_any_name:ident,
        $fallback_any_name:ident,
    ) => {
        #[doc = concat!("`", stringify!($t), "` ", $desc)]
        pub fn $const_name<const DIMS: usize>(a: &[$t]) -> $t {
            assert_eq!(a.len(), DIMS, "Provided vector does not match size DIMS");

            unsafe {
                crate::dispatch!(
                    avx512 = $avx512_const_name::<DIMS> => (a)
                    avx2fma = $avx2fma_const_name::<DIMS> => (a)
                    avx2 = $avx2_const_name::<DIMS> => (a)
                    neon = $neon_const_name::<DIMS> => (a)
                    fallback = $fallback_const_name::<DIMS> => (a)
                )
            }
        }

        #[doc = concat!("`", stringify!($t), "` ", $desc)]
        pub fn $any_name(a: &[$t]) -> $t {
            unsafe {
                crate::dispatch!(
                    avx512 = $avx512_any_name => (a)
                    avx2fma = $avx2fma_any_name => (a)
                    avx2 = $avx2_any_name => (a)
                    neon = $neon_any_name => (a)
                    fallback = $fallback_any_name => (a)
                )
            }
        }
    };
}

macro_rules! export_safe_nofma_norm_op {
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
                    avx512 = $avx512_const_name::<DIMS> => (a)
                    avx2 = $avx2_const_name::<DIMS> => (a)
                    neon = $neon_const_name::<DIMS> => (a)
                    fallback = $fallback_const_name::<DIMS> => (a)
                )
            }
        }

        #[doc = concat!("`", stringify!($t), "` ", $desc)]
        pub fn $any_name(a: &[$t]) -> $t {
            unsafe {
                crate::dispatch!(
                    avx512 = $avx512_any_name => (a)
                    avx2 = $avx2_any_name => (a)
                    neon = $neon_any_name => (a)
                    fallback = $fallback_any_name => (a)
                )
            }
        }
    };
}

export_safe_fma_norm_op!(
    description = "Calculates the squared L2 norm of the provided vector",
    ty = f32,
    const_name = f32_xconst_squared_norm,
    any_name = f32_xany_squared_norm,
    f32_xconst_avx512_fma_squared_norm,
    f32_xconst_avx2_fma_squared_norm,
    f32_xconst_avx2_nofma_squared_norm,
    f32_xconst_neon_fma_squared_norm,
    f32_xconst_fallback_nofma_squared_norm,
    f32_xany_avx512_fma_squared_norm,
    f32_xany_avx2_fma_squared_norm,
    f32_xany_avx2_nofma_squared_norm,
    f32_xany_neon_fma_squared_norm,
    f32_xany_fallback_nofma_squared_norm,
);

export_safe_fma_norm_op!(
    description = "Calculates the squared L2 norm of the provided vector",
    ty = f64,
    const_name = f64_xconst_squared_norm,
    any_name = f64_xany_squared_norm,
    f64_xconst_avx512_fma_squared_norm,
    f64_xconst_avx2_fma_squared_norm,
    f64_xconst_avx2_nofma_squared_norm,
    f64_xconst_neon_fma_squared_norm,
    f64_xconst_fallback_nofma_squared_norm,
    f64_xany_avx512_fma_squared_norm,
    f64_xany_avx2_fma_squared_norm,
    f64_xany_avx2_nofma_squared_norm,
    f64_xany_neon_fma_squared_norm,
    f64_xany_fallback_nofma_squared_norm,
);

export_safe_nofma_norm_op!(
    description = "Calculates the squared L2 norm of the provided vector",
    ty = u8,
    const_name = u8_xconst_squared_norm,
    any_name = u8_xany_squared_norm,
    u8_xconst_avx512_nofma_squared_norm,
    u8_xconst_avx2_nofma_squared_norm,
    u8_xconst_neon_nofma_squared_norm,
    u8_xconst_fallback_nofma_squared_norm,
    u8_xany_avx512_nofma_squared_norm,
    u8_xany_avx2_nofma_squared_norm,
    u8_xany_neon_nofma_squared_norm,
    u8_xany_fallback_nofma_squared_norm,
);

export_safe_nofma_norm_op!(
    description = "Calculates the squared L2 norm of the provided vector",
    ty = u16,
    const_name = u16_xconst_squared_norm,
    any_name = u16_xany_squared_norm,
    u16_xconst_avx512_nofma_squared_norm,
    u16_xconst_avx2_nofma_squared_norm,
    u16_xconst_neon_nofma_squared_norm,
    u16_xconst_fallback_nofma_squared_norm,
    u16_xany_avx512_nofma_squared_norm,
    u16_xany_avx2_nofma_squared_norm,
    u16_xany_neon_nofma_squared_norm,
    u16_xany_fallback_nofma_squared_norm,
);

export_safe_nofma_norm_op!(
    description = "Calculates the squared L2 norm of the provided vector",
    ty = u32,
    const_name = u32_xconst_squared_norm,
    any_name = u32_xany_squared_norm,
    u32_xconst_avx512_nofma_squared_norm,
    u32_xconst_avx2_nofma_squared_norm,
    u32_xconst_neon_nofma_squared_norm,
    u32_xconst_fallback_nofma_squared_norm,
    u32_xany_avx512_nofma_squared_norm,
    u32_xany_avx2_nofma_squared_norm,
    u32_xany_neon_nofma_squared_norm,
    u32_xany_fallback_nofma_squared_norm,
);

export_safe_nofma_norm_op!(
    description = "Calculates the squared L2 norm of the provided vector",
    ty = u64,
    const_name = u64_xconst_squared_norm,
    any_name = u64_xany_squared_norm,
    u64_xconst_avx512_nofma_squared_norm,
    u64_xconst_avx2_nofma_squared_norm,
    u64_xconst_neon_nofma_squared_norm,
    u64_xconst_fallback_nofma_squared_norm,
    u64_xany_avx512_nofma_squared_norm,
    u64_xany_avx2_nofma_squared_norm,
    u64_xany_neon_nofma_squared_norm,
    u64_xany_fallback_nofma_squared_norm,
);

export_safe_nofma_norm_op!(
    description = "Calculates the squared L2 norm of the provided vector",
    ty = i8,
    const_name = i8_xconst_squared_norm,
    any_name = i8_xany_squared_norm,
    i8_xconst_avx512_nofma_squared_norm,
    i8_xconst_avx2_nofma_squared_norm,
    i8_xconst_neon_nofma_squared_norm,
    i8_xconst_fallback_nofma_squared_norm,
    i8_xany_avx512_nofma_squared_norm,
    i8_xany_avx2_nofma_squared_norm,
    i8_xany_neon_nofma_squared_norm,
    i8_xany_fallback_nofma_squared_norm,
);

export_safe_nofma_norm_op!(
    description = "Calculates the squared L2 norm of the provided vector",
    ty = i16,
    const_name = i16_xconst_squared_norm,
    any_name = i16_xany_squared_norm,
    i16_xconst_avx512_nofma_squared_norm,
    i16_xconst_avx2_nofma_squared_norm,
    i16_xconst_neon_nofma_squared_norm,
    i16_xconst_fallback_nofma_squared_norm,
    i16_xany_avx512_nofma_squared_norm,
    i16_xany_avx2_nofma_squared_norm,
    i16_xany_neon_nofma_squared_norm,
    i16_xany_fallback_nofma_squared_norm,
);

export_safe_nofma_norm_op!(
    description = "Calculates the squared L2 norm of the provided vector",
    ty = i32,
    const_name = i32_xconst_squared_norm,
    any_name = i32_xany_squared_norm,
    i32_xconst_avx512_nofma_squared_norm,
    i32_xconst_avx2_nofma_squared_norm,
    i32_xconst_neon_nofma_squared_norm,
    i32_xconst_fallback_nofma_squared_norm,
    i32_xany_avx512_nofma_squared_norm,
    i32_xany_avx2_nofma_squared_norm,
    i32_xany_neon_nofma_squared_norm,
    i32_xany_fallback_nofma_squared_norm,
);

export_safe_nofma_norm_op!(
    description = "Calculates the squared L2 norm of the provided vector",
    ty = i64,
    const_name = i64_xconst_squared_norm,
    any_name = i64_xany_squared_norm,
    i64_xconst_avx512_nofma_squared_norm,
    i64_xconst_avx2_nofma_squared_norm,
    i64_xconst_neon_nofma_squared_norm,
    i64_xconst_fallback_nofma_squared_norm,
    i64_xany_avx512_nofma_squared_norm,
    i64_xany_avx2_nofma_squared_norm,
    i64_xany_neon_nofma_squared_norm,
    i64_xany_fallback_nofma_squared_norm,
);
