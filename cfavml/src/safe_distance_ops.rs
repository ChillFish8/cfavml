//! Vector space distance operations.
//!
//! These exported methods are safe to call and select the fastest available instruction set
//! to use at runtime.
//!
//! Both `xconst` and `xany` variants of each operation are provided.
//!
//! The following distance operations are provided:
//!
//! - Cosine
//! - Squared Euclidean
//! - Dot Product
//!
//! ```
//! use cfavml::*;
//!
//! const DIMS: usize = 3;
//! let a = [1.0, 2.0, 3.0];
//! let b = [1.0, 2.0, 3.0];
//!
//! assert_eq!(f32_xany_dot(&a, &b), 14.0);
//! assert_eq!(f32_xany_cosine(&a, &b), 0.0);
//! assert_eq!(f32_xany_squared_euclidean(&a, &b), 0.0);
//!
//! assert_eq!(f32_xconst_dot::<DIMS>(&a, &b), 14.0);
//! assert_eq!(f32_xconst_cosine::<DIMS>(&a, &b), 0.0);
//! assert_eq!(f32_xconst_squared_euclidean::<DIMS>(&a, &b), 0.0);
//! ```
use crate::danger::*;

macro_rules! export_safe_distance_op {
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
        pub fn $const_name<const DIMS: usize>(a: &[$t], b: &[$t]) -> $t {
            assert_eq!(a.len(), b.len(), "Input vector sizes do not match");

            unsafe {
                crate::dispatch!(
                    avx512 = $avx512_const_name::<DIMS>,
                    avx2fma = $avx2fma_const_name::<DIMS>,
                    avx2 = $avx2_const_name::<DIMS>,
                    neon = $neon_const_name::<DIMS>,
                    fallback = $fallback_const_name::<DIMS>,
                    args = (a, b)
                )
            }
        }

        #[doc = concat!("`", stringify!($t), "` ", $desc)]
        pub fn $any_name(a: &[$t], b: &[$t]) -> $t {
            assert_eq!(a.len(), b.len(), "Input vector sizes do not match");

            unsafe {
                crate::dispatch!(
                    avx512 = $avx512_any_name,
                    avx2fma = $avx2fma_any_name,
                    avx2 = $avx2_any_name,
                    neon = $neon_any_name,
                    fallback = $fallback_any_name,
                    args = (a, b)
                )
            }
        }
    };
}

export_safe_distance_op!(
    description = "Dot product of two provided vectors",
    ty = f32,
    const_name = f32_xconst_dot,
    any_name = f32_xany_dot,
    f32_xconst_avx512_fma_dot,
    f32_xconst_avx2_fma_dot,
    f32_xconst_avx2_nofma_dot,
    f32_xconst_neon_fma_dot,
    f32_xconst_fallback_nofma_dot,
    f32_xany_avx512_fma_dot,
    f32_xany_avx2_fma_dot,
    f32_xany_avx2_nofma_dot,
    f32_xany_neon_fma_dot,
    f32_xany_fallback_nofma_dot,
);
export_safe_distance_op!(
    description = "Cosine distance of two provided vectors",
    ty = f32,
    const_name = f32_xconst_cosine,
    any_name = f32_xany_cosine,
    f32_xconst_avx512_fma_cosine,
    f32_xconst_avx2_fma_cosine,
    f32_xconst_avx2_nofma_cosine,
    f32_xconst_neon_fma_cosine,
    f32_xconst_fallback_nofma_cosine,
    f32_xany_avx512_fma_cosine,
    f32_xany_avx2_fma_cosine,
    f32_xany_avx2_nofma_cosine,
    f32_xany_neon_fma_cosine,
    f32_xany_fallback_nofma_cosine,
);
export_safe_distance_op!(
    description = "Squared Euclidean distance of two provided vectors",
    ty = f32,
    const_name = f32_xconst_squared_euclidean,
    any_name = f32_xany_squared_euclidean,
    f32_xconst_avx512_fma_squared_euclidean,
    f32_xconst_avx2_fma_squared_euclidean,
    f32_xconst_avx2_nofma_squared_euclidean,
    f32_xconst_neon_fma_squared_euclidean,
    f32_xconst_fallback_nofma_squared_euclidean,
    f32_xany_avx512_fma_squared_euclidean,
    f32_xany_avx2_fma_squared_euclidean,
    f32_xany_avx2_nofma_squared_euclidean,
    f32_xany_neon_fma_squared_euclidean,
    f32_xany_fallback_nofma_squared_euclidean,
);

export_safe_distance_op!(
    description = "Dot product of two provided vectors",
    ty = f64,
    const_name = f64_xconst_dot,
    any_name = f64_xany_dot,
    f64_xconst_avx512_fma_dot,
    f64_xconst_avx2_fma_dot,
    f64_xconst_avx2_nofma_dot,
    f64_xconst_neon_fma_dot,
    f64_xconst_fallback_nofma_dot,
    f64_xany_avx512_fma_dot,
    f64_xany_avx2_fma_dot,
    f64_xany_avx2_nofma_dot,
    f64_xany_neon_fma_dot,
    f64_xany_fallback_nofma_dot,
);
export_safe_distance_op!(
    description = "Cosine distance of two provided vectors",
    ty = f64,
    const_name = f64_xconst_cosine,
    any_name = f64_xany_cosine,
    f64_xconst_avx512_fma_cosine,
    f64_xconst_avx2_fma_cosine,
    f64_xconst_avx2_nofma_cosine,
    f64_xconst_neon_fma_cosine,
    f64_xconst_fallback_nofma_cosine,
    f64_xany_avx512_fma_cosine,
    f64_xany_avx2_fma_cosine,
    f64_xany_avx2_nofma_cosine,
    f64_xany_neon_fma_cosine,
    f64_xany_fallback_nofma_cosine,
);
export_safe_distance_op!(
    description = "Squared Euclidean distance of two provided vectors",
    ty = f64,
    const_name = f64_xconst_squared_euclidean,
    any_name = f64_xany_squared_euclidean,
    f64_xconst_avx512_fma_squared_euclidean,
    f64_xconst_avx2_fma_squared_euclidean,
    f64_xconst_avx2_nofma_squared_euclidean,
    f64_xconst_neon_fma_squared_euclidean,
    f64_xconst_fallback_nofma_squared_euclidean,
    f64_xany_avx512_fma_squared_euclidean,
    f64_xany_avx2_fma_squared_euclidean,
    f64_xany_avx2_nofma_squared_euclidean,
    f64_xany_neon_fma_squared_euclidean,
    f64_xany_fallback_nofma_squared_euclidean,
);

export_safe_distance_op!(
    description = "Dot product of two provided vectors",
    ty = u8,
    const_name = u8_xconst_dot,
    any_name = u8_xany_dot,
    u8_xconst_avx512_nofma_dot,
    u8_xconst_avx2_nofma_dot,
    u8_xconst_avx2_nofma_dot,
    u8_xconst_neon_nofma_dot,
    u8_xconst_fallback_nofma_dot,
    u8_xany_avx512_nofma_dot,
    u8_xany_avx2_nofma_dot,
    u8_xany_avx2_nofma_dot,
    u8_xany_neon_nofma_dot,
    u8_xany_fallback_nofma_dot,
);
export_safe_distance_op!(
    description = "Cosine distance of two provided vectors",
    ty = u8,
    const_name = u8_xconst_cosine,
    any_name = u8_xany_cosine,
    u8_xconst_avx512_nofma_cosine,
    u8_xconst_avx2_nofma_cosine,
    u8_xconst_avx2_nofma_cosine,
    u8_xconst_neon_nofma_cosine,
    u8_xconst_fallback_nofma_cosine,
    u8_xany_avx512_nofma_cosine,
    u8_xany_avx2_nofma_cosine,
    u8_xany_avx2_nofma_cosine,
    u8_xany_neon_nofma_cosine,
    u8_xany_fallback_nofma_cosine,
);
export_safe_distance_op!(
    description = "Squared Euclidean distance of two provided vectors",
    ty = u8,
    const_name = u8_xconst_squared_euclidean,
    any_name = u8_xany_squared_euclidean,
    u8_xconst_avx512_nofma_squared_euclidean,
    u8_xconst_avx2_nofma_squared_euclidean,
    u8_xconst_avx2_nofma_squared_euclidean,
    u8_xconst_neon_nofma_squared_euclidean,
    u8_xconst_fallback_nofma_squared_euclidean,
    u8_xany_avx512_nofma_squared_euclidean,
    u8_xany_avx2_nofma_squared_euclidean,
    u8_xany_avx2_nofma_squared_euclidean,
    u8_xany_neon_nofma_squared_euclidean,
    u8_xany_fallback_nofma_squared_euclidean,
);

export_safe_distance_op!(
    description = "Dot product of two provided vectors",
    ty = u16,
    const_name = u16_xconst_dot,
    any_name = u16_xany_dot,
    u16_xconst_avx512_nofma_dot,
    u16_xconst_avx2_nofma_dot,
    u16_xconst_avx2_nofma_dot,
    u16_xconst_neon_nofma_dot,
    u16_xconst_fallback_nofma_dot,
    u16_xany_avx512_nofma_dot,
    u16_xany_avx2_nofma_dot,
    u16_xany_avx2_nofma_dot,
    u16_xany_neon_nofma_dot,
    u16_xany_fallback_nofma_dot,
);
export_safe_distance_op!(
    description = "Cosine distance of two provided vectors",
    ty = u16,
    const_name = u16_xconst_cosine,
    any_name = u16_xany_cosine,
    u16_xconst_avx512_nofma_cosine,
    u16_xconst_avx2_nofma_cosine,
    u16_xconst_avx2_nofma_cosine,
    u16_xconst_neon_nofma_cosine,
    u16_xconst_fallback_nofma_cosine,
    u16_xany_avx512_nofma_cosine,
    u16_xany_avx2_nofma_cosine,
    u16_xany_avx2_nofma_cosine,
    u16_xany_neon_nofma_cosine,
    u16_xany_fallback_nofma_cosine,
);
export_safe_distance_op!(
    description = "Squared Euclidean distance of two provided vectors",
    ty = u16,
    const_name = u16_xconst_squared_euclidean,
    any_name = u16_xany_squared_euclidean,
    u16_xconst_avx512_nofma_squared_euclidean,
    u16_xconst_avx2_nofma_squared_euclidean,
    u16_xconst_avx2_nofma_squared_euclidean,
    u16_xconst_neon_nofma_squared_euclidean,
    u16_xconst_fallback_nofma_squared_euclidean,
    u16_xany_avx512_nofma_squared_euclidean,
    u16_xany_avx2_nofma_squared_euclidean,
    u16_xany_avx2_nofma_squared_euclidean,
    u16_xany_neon_nofma_squared_euclidean,
    u16_xany_fallback_nofma_squared_euclidean,
);

export_safe_distance_op!(
    description = "Dot product of two provided vectors",
    ty = u32,
    const_name = u32_xconst_dot,
    any_name = u32_xany_dot,
    u32_xconst_avx512_nofma_dot,
    u32_xconst_avx2_nofma_dot,
    u32_xconst_avx2_nofma_dot,
    u32_xconst_neon_nofma_dot,
    u32_xconst_fallback_nofma_dot,
    u32_xany_avx512_nofma_dot,
    u32_xany_avx2_nofma_dot,
    u32_xany_avx2_nofma_dot,
    u32_xany_neon_nofma_dot,
    u32_xany_fallback_nofma_dot,
);
export_safe_distance_op!(
    description = "Cosine distance of two provided vectors",
    ty = u32,
    const_name = u32_xconst_cosine,
    any_name = u32_xany_cosine,
    u32_xconst_avx512_nofma_cosine,
    u32_xconst_avx2_nofma_cosine,
    u32_xconst_avx2_nofma_cosine,
    u32_xconst_neon_nofma_cosine,
    u32_xconst_fallback_nofma_cosine,
    u32_xany_avx512_nofma_cosine,
    u32_xany_avx2_nofma_cosine,
    u32_xany_avx2_nofma_cosine,
    u32_xany_neon_nofma_cosine,
    u32_xany_fallback_nofma_cosine,
);
export_safe_distance_op!(
    description = "Squared Euclidean distance of two provided vectors",
    ty = u32,
    const_name = u32_xconst_squared_euclidean,
    any_name = u32_xany_squared_euclidean,
    u32_xconst_avx512_nofma_squared_euclidean,
    u32_xconst_avx2_nofma_squared_euclidean,
    u32_xconst_avx2_nofma_squared_euclidean,
    u32_xconst_neon_nofma_squared_euclidean,
    u32_xconst_fallback_nofma_squared_euclidean,
    u32_xany_avx512_nofma_squared_euclidean,
    u32_xany_avx2_nofma_squared_euclidean,
    u32_xany_avx2_nofma_squared_euclidean,
    u32_xany_neon_nofma_squared_euclidean,
    u32_xany_fallback_nofma_squared_euclidean,
);

export_safe_distance_op!(
    description = "Dot product of two provided vectors",
    ty = u64,
    const_name = u64_xconst_dot,
    any_name = u64_xany_dot,
    u64_xconst_avx512_nofma_dot,
    u64_xconst_avx2_nofma_dot,
    u64_xconst_avx2_nofma_dot,
    u64_xconst_neon_nofma_dot,
    u64_xconst_fallback_nofma_dot,
    u64_xany_avx512_nofma_dot,
    u64_xany_avx2_nofma_dot,
    u64_xany_avx2_nofma_dot,
    u64_xany_neon_nofma_dot,
    u64_xany_fallback_nofma_dot,
);
export_safe_distance_op!(
    description = "Cosine distance of two provided vectors",
    ty = u64,
    const_name = u64_xconst_cosine,
    any_name = u64_xany_cosine,
    u64_xconst_avx512_nofma_cosine,
    u64_xconst_avx2_nofma_cosine,
    u64_xconst_avx2_nofma_cosine,
    u64_xconst_neon_nofma_cosine,
    u64_xconst_fallback_nofma_cosine,
    u64_xany_avx512_nofma_cosine,
    u64_xany_avx2_nofma_cosine,
    u64_xany_avx2_nofma_cosine,
    u64_xany_neon_nofma_cosine,
    u64_xany_fallback_nofma_cosine,
);
export_safe_distance_op!(
    description = "Squared Euclidean distance of two provided vectors",
    ty = u64,
    const_name = u64_xconst_squared_euclidean,
    any_name = u64_xany_squared_euclidean,
    u64_xconst_avx512_nofma_squared_euclidean,
    u64_xconst_avx2_nofma_squared_euclidean,
    u64_xconst_avx2_nofma_squared_euclidean,
    u64_xconst_neon_nofma_squared_euclidean,
    u64_xconst_fallback_nofma_squared_euclidean,
    u64_xany_avx512_nofma_squared_euclidean,
    u64_xany_avx2_nofma_squared_euclidean,
    u64_xany_avx2_nofma_squared_euclidean,
    u64_xany_neon_nofma_squared_euclidean,
    u64_xany_fallback_nofma_squared_euclidean,
);

export_safe_distance_op!(
    description = "Dot product of two provided vectors",
    ty = i8,
    const_name = i8_xconst_dot,
    any_name = i8_xany_dot,
    i8_xconst_avx512_nofma_dot,
    i8_xconst_avx2_nofma_dot,
    i8_xconst_avx2_nofma_dot,
    i8_xconst_neon_nofma_dot,
    i8_xconst_fallback_nofma_dot,
    i8_xany_avx512_nofma_dot,
    i8_xany_avx2_nofma_dot,
    i8_xany_avx2_nofma_dot,
    i8_xany_neon_nofma_dot,
    i8_xany_fallback_nofma_dot,
);
export_safe_distance_op!(
    description = "Cosine distance of two provided vectors",
    ty = i8,
    const_name = i8_xconst_cosine,
    any_name = i8_xany_cosine,
    i8_xconst_avx512_nofma_cosine,
    i8_xconst_avx2_nofma_cosine,
    i8_xconst_avx2_nofma_cosine,
    i8_xconst_neon_nofma_cosine,
    i8_xconst_fallback_nofma_cosine,
    i8_xany_avx512_nofma_cosine,
    i8_xany_avx2_nofma_cosine,
    i8_xany_avx2_nofma_cosine,
    i8_xany_neon_nofma_cosine,
    i8_xany_fallback_nofma_cosine,
);
export_safe_distance_op!(
    description = "Squared Euclidean distance of two provided vectors",
    ty = i8,
    const_name = i8_xconst_squared_euclidean,
    any_name = i8_xany_squared_euclidean,
    i8_xconst_avx512_nofma_squared_euclidean,
    i8_xconst_avx2_nofma_squared_euclidean,
    i8_xconst_avx2_nofma_squared_euclidean,
    i8_xconst_neon_nofma_squared_euclidean,
    i8_xconst_fallback_nofma_squared_euclidean,
    i8_xany_avx512_nofma_squared_euclidean,
    i8_xany_avx2_nofma_squared_euclidean,
    i8_xany_avx2_nofma_squared_euclidean,
    i8_xany_neon_nofma_squared_euclidean,
    i8_xany_fallback_nofma_squared_euclidean,
);

export_safe_distance_op!(
    description = "Dot product of two provided vectors",
    ty = i16,
    const_name = i16_xconst_dot,
    any_name = i16_xany_dot,
    i16_xconst_avx512_nofma_dot,
    i16_xconst_avx2_nofma_dot,
    i16_xconst_avx2_nofma_dot,
    i16_xconst_neon_nofma_dot,
    i16_xconst_fallback_nofma_dot,
    i16_xany_avx512_nofma_dot,
    i16_xany_avx2_nofma_dot,
    i16_xany_avx2_nofma_dot,
    i16_xany_neon_nofma_dot,
    i16_xany_fallback_nofma_dot,
);
export_safe_distance_op!(
    description = "Cosine distance of two provided vectors",
    ty = i16,
    const_name = i16_xconst_cosine,
    any_name = i16_xany_cosine,
    i16_xconst_avx512_nofma_cosine,
    i16_xconst_avx2_nofma_cosine,
    i16_xconst_avx2_nofma_cosine,
    i16_xconst_neon_nofma_cosine,
    i16_xconst_fallback_nofma_cosine,
    i16_xany_avx512_nofma_cosine,
    i16_xany_avx2_nofma_cosine,
    i16_xany_avx2_nofma_cosine,
    i16_xany_neon_nofma_cosine,
    i16_xany_fallback_nofma_cosine,
);
export_safe_distance_op!(
    description = "Squared Euclidean distance of two provided vectors",
    ty = i16,
    const_name = i16_xconst_squared_euclidean,
    any_name = i16_xany_squared_euclidean,
    i16_xconst_avx512_nofma_squared_euclidean,
    i16_xconst_avx2_nofma_squared_euclidean,
    i16_xconst_avx2_nofma_squared_euclidean,
    i16_xconst_neon_nofma_squared_euclidean,
    i16_xconst_fallback_nofma_squared_euclidean,
    i16_xany_avx512_nofma_squared_euclidean,
    i16_xany_avx2_nofma_squared_euclidean,
    i16_xany_avx2_nofma_squared_euclidean,
    i16_xany_neon_nofma_squared_euclidean,
    i16_xany_fallback_nofma_squared_euclidean,
);

export_safe_distance_op!(
    description = "Dot product of two provided vectors",
    ty = i32,
    const_name = i32_xconst_dot,
    any_name = i32_xany_dot,
    i32_xconst_avx512_nofma_dot,
    i32_xconst_avx2_nofma_dot,
    i32_xconst_avx2_nofma_dot,
    i32_xconst_neon_nofma_dot,
    i32_xconst_fallback_nofma_dot,
    i32_xany_avx512_nofma_dot,
    i32_xany_avx2_nofma_dot,
    i32_xany_avx2_nofma_dot,
    i32_xany_neon_nofma_dot,
    i32_xany_fallback_nofma_dot,
);
export_safe_distance_op!(
    description = "Cosine distance of two provided vectors",
    ty = i32,
    const_name = i32_xconst_cosine,
    any_name = i32_xany_cosine,
    i32_xconst_avx512_nofma_cosine,
    i32_xconst_avx2_nofma_cosine,
    i32_xconst_avx2_nofma_cosine,
    i32_xconst_neon_nofma_cosine,
    i32_xconst_fallback_nofma_cosine,
    i32_xany_avx512_nofma_cosine,
    i32_xany_avx2_nofma_cosine,
    i32_xany_avx2_nofma_cosine,
    i32_xany_neon_nofma_cosine,
    i32_xany_fallback_nofma_cosine,
);
export_safe_distance_op!(
    description = "Squared Euclidean distance of two provided vectors",
    ty = i32,
    const_name = i32_xconst_squared_euclidean,
    any_name = i32_xany_squared_euclidean,
    i32_xconst_avx512_nofma_squared_euclidean,
    i32_xconst_avx2_nofma_squared_euclidean,
    i32_xconst_avx2_nofma_squared_euclidean,
    i32_xconst_neon_nofma_squared_euclidean,
    i32_xconst_fallback_nofma_squared_euclidean,
    i32_xany_avx512_nofma_squared_euclidean,
    i32_xany_avx2_nofma_squared_euclidean,
    i32_xany_avx2_nofma_squared_euclidean,
    i32_xany_neon_nofma_squared_euclidean,
    i32_xany_fallback_nofma_squared_euclidean,
);

export_safe_distance_op!(
    description = "Dot product of two provided vectors",
    ty = i64,
    const_name = i64_xconst_dot,
    any_name = i64_xany_dot,
    i64_xconst_avx512_nofma_dot,
    i64_xconst_avx2_nofma_dot,
    i64_xconst_avx2_nofma_dot,
    i64_xconst_neon_nofma_dot,
    i64_xconst_fallback_nofma_dot,
    i64_xany_avx512_nofma_dot,
    i64_xany_avx2_nofma_dot,
    i64_xany_avx2_nofma_dot,
    i64_xany_neon_nofma_dot,
    i64_xany_fallback_nofma_dot,
);
export_safe_distance_op!(
    description = "Cosine distance of two provided vectors",
    ty = i64,
    const_name = i64_xconst_cosine,
    any_name = i64_xany_cosine,
    i64_xconst_avx512_nofma_cosine,
    i64_xconst_avx2_nofma_cosine,
    i64_xconst_avx2_nofma_cosine,
    i64_xconst_neon_nofma_cosine,
    i64_xconst_fallback_nofma_cosine,
    i64_xany_avx512_nofma_cosine,
    i64_xany_avx2_nofma_cosine,
    i64_xany_avx2_nofma_cosine,
    i64_xany_neon_nofma_cosine,
    i64_xany_fallback_nofma_cosine,
);
export_safe_distance_op!(
    description = "Squared Euclidean distance of two provided vectors",
    ty = i64,
    const_name = i64_xconst_squared_euclidean,
    any_name = i64_xany_squared_euclidean,
    i64_xconst_avx512_nofma_squared_euclidean,
    i64_xconst_avx2_nofma_squared_euclidean,
    i64_xconst_avx2_nofma_squared_euclidean,
    i64_xconst_neon_nofma_squared_euclidean,
    i64_xconst_fallback_nofma_squared_euclidean,
    i64_xany_avx512_nofma_squared_euclidean,
    i64_xany_avx2_nofma_squared_euclidean,
    i64_xany_avx2_nofma_squared_euclidean,
    i64_xany_neon_nofma_squared_euclidean,
    i64_xany_fallback_nofma_squared_euclidean,
);

#[cfg(test)]
/// Tests the exposed safe API.
///
/// These are more sanity checks than anything else, because they can change depending
/// on the system running them (as they are runtime selected)
mod tests {
    use super::*;

    macro_rules! test_distance {
        ($t:ident) => {
            paste::paste! {
                #[test]
                fn [< test_ $t _dot_product >]() {
                    let l1 = [1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t];
                    let l2 = [1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t];

                    let res = [<$t _xany_dot >](&l1, &l2);

                    let expected = crate::test_utils::simple_dot(&l1, &l2);
                    assert_eq!(res, expected, "Dot product op miss-match");
                }

                #[test]
                fn [< test_ $t _cosine >]() {
                    let l1 = [1 as $t, 2 as $t, 1 as $t, 1 as $t, 2 as $t];
                    let l2 = [1 as $t, 2 as $t, 2 as $t, 4 as $t, 2 as $t];

                    let res = [<$t _xany_cosine >](&l1, &l2);

                    let expected = crate::test_utils::simple_cosine(&l1, &l2);
                    assert_eq!(res, expected, "Cosine op miss-match");
                }

                #[test]
                fn [< test_ $t _euclidean >]() {
                    let l1 = [1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t];
                    let l2 = [1 as $t, 2 as $t, 3 as $t, 4 as $t, 5 as $t];

                    let res = [<$t _xany_squared_euclidean >](&l1, &l2);

                    let expected = crate::test_utils::simple_euclidean(&l1, &l2);
                    assert_eq!(res, expected, "Euclidean op miss-match");
                }
            }
        };
    }

    test_distance!(f32);
    test_distance!(f64);
    test_distance!(u8);
    test_distance!(u16);
    test_distance!(u32);
    test_distance!(u64);
    test_distance!(i8);
    test_distance!(i16);
    test_distance!(i32);
    test_distance!(i64);
}
