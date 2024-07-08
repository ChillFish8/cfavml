//! Generated non-generic methods for x86 based SIMD operations.
//!
//! Although the generic methods can provide greater ergonomics within your code
//! they do not apply the necessary feature assumptions or type ergonomics, that
//! is the purpose of these exported methods which ultimately end up calling the
//! generic variant.
//!
//! Method names are exported in the following format:
//! ```no_run
//! <type>_x<const/any>_<arch>_<fma/nofma>_<op>
//! ```
//!
//! ### `xconst` vs `xany`
//!
//! We export const generic versions of the impls which allow the compiler to perform further
//! optimizations on the routines like unrolling and tail loop elimination, if you are using
//! these routines where you know the vector dimensions at compile time, it is probably worth
//! using the `xconst` variants, otherwise use the `xany` methods.
//!

use super::*;
use crate::math::AutoMath;

macro_rules! export_distance_op {
    (
        description = $desc:expr,
        ty = $t:ident,
        register = $im:ident,
        op = $op:ident,
        xconst = $xconst_name:ident,
        xany = $xany_name:ident,
    ) => {
        #[doc = concat!("`", stringify!($t), "` ", $desc)]
        ///
        /// # Safety
        ///
        /// Vectors **`a`** and **`b`** must be equal in length to that specified in **`DIMS`**,
        /// otherwise out of bound access can occur.
        pub unsafe fn $xconst_name<const DIMS: usize>(a: &[$t], b: &[$t]) -> $t {
            $op::<_, $im, AutoMath>(DIMS, a, b)
        }

        #[doc = concat!("`", stringify!($t), "` ", $desc)]
        ///
        /// # Safety
        ///
        /// Vectors **`a`** and **`b`** must be equal in length otherwise out of bound
        /// access can occur.
        pub unsafe fn $xany_name(a: &[$t], b: &[$t]) -> $t {
            $op::<_, $im, AutoMath>(a.len(), a, b)
        }
    };
    (
        description = $desc:expr,
        ty = $t:ident,
        register = $im:ident,
        op = $op:ident,
        xconst = $xconst_name:ident,
        xany = $xany_name:ident,
        features = $($feat:expr $(,)?)*
    ) => {
        #[target_feature($(enable = $feat , )*)]
        #[doc = concat!("`", stringify!($t), "` ", $desc)]
        #[doc = "\n\n# Safety\n\n"]
        #[doc = concat!("The following CPU flags must be available: ", $("**", $feat, "**", ", ",)*)]
        ///
        /// Vectors **`a`** and **`b`** must be equal in length to that specified in **`DIMS`**,
        /// otherwise out of bound access can occur.
        pub unsafe fn $xconst_name<const DIMS: usize>(a: &[$t], b: &[$t]) -> $t {
            $op::<_, $im, AutoMath>(DIMS, a, b)
        }

        #[target_feature($(enable = $feat , )*)]
        #[doc = concat!("`", stringify!($t), "` ", $desc)]
        #[doc = "\n\n# Safety\n\n"]
        #[doc = concat!("The following CPU flags must be available: ", $("**", $feat, "**", ", ",)*)]
        ///
        /// Vectors **`a`** and **`b`** must be equal in length otherwise out of bound
        /// access can occur.
        pub unsafe fn $xany_name(a: &[$t], b: &[$t]) -> $t {
            $op::<_, $im, AutoMath>(a.len(), a, b)
        }
    };
}

/// Vector space distance ops on the fallback implementations.
///
/// These methods do not strictly require any CPU feature but do auto-vectorize
/// well when target features are explicitly provided and can be used to
/// provide accelerated operations on non-explicitly supported SIMD architectures.
pub mod distance_ops_fallback {
    use super::*;

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = i8,
        register = Fallback,
        op = generic_dot_product,
        xconst = i8_xconst_fallback_nofma_dot,
        xany = i8_xany_fallback_nofma_dot,
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = i8,
        register = Fallback,
        op = generic_cosine,
        xconst = i8_xconst_fallback_nofma_cosine,
        xany = i8_xany_fallback_nofma_cosine,
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = i8,
        register = Fallback,
        op = generic_euclidean,
        xconst = i8_xconst_fallback_nofma_squared_euclidean,
        xany = i8_xany_fallback_nofma_squared_euclidean,
    );

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = i16,
        register = Fallback,
        op = generic_dot_product,
        xconst = i16_xconst_fallback_nofma_dot,
        xany = i16_xany_fallback_nofma_dot,
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = i16,
        register = Fallback,
        op = generic_cosine,
        xconst = i16_xconst_fallback_nofma_cosine,
        xany = i16_xany_fallback_nofma_cosine,
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = i16,
        register = Fallback,
        op = generic_euclidean,
        xconst = i16_xconst_fallback_nofma_squared_euclidean,
        xany = i16_xany_fallback_nofma_squared_euclidean,
    );

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = i32,
        register = Fallback,
        op = generic_dot_product,
        xconst = i32_xconst_fallback_nofma_dot,
        xany = i32_xany_fallback_nofma_dot,
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = i32,
        register = Fallback,
        op = generic_cosine,
        xconst = i32_xconst_fallback_nofma_cosine,
        xany = i32_xany_fallback_nofma_cosine,
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = i32,
        register = Fallback,
        op = generic_euclidean,
        xconst = i32_xconst_fallback_nofma_squared_euclidean,
        xany = i32_xany_fallback_nofma_squared_euclidean,
    );

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = i64,
        register = Fallback,
        op = generic_dot_product,
        xconst = i64_xconst_fallback_nofma_dot,
        xany = i64_xany_fallback_nofma_dot,
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = i64,
        register = Fallback,
        op = generic_cosine,
        xconst = i64_xconst_fallback_nofma_cosine,
        xany = i64_xany_fallback_nofma_cosine,
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = i64,
        register = Fallback,
        op = generic_euclidean,
        xconst = i64_xconst_fallback_nofma_squared_euclidean,
        xany = i64_xany_fallback_nofma_squared_euclidean,
    );

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = u8,
        register = Fallback,
        op = generic_dot_product,
        xconst = u8_xconst_fallback_nofma_dot,
        xany = u8_xany_fallback_nofma_dot,
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = u8,
        register = Fallback,
        op = generic_cosine,
        xconst = u8_xconst_fallback_nofma_cosine,
        xany = u8_xany_fallback_nofma_cosine,
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = u8,
        register = Fallback,
        op = generic_euclidean,
        xconst = u8_xconst_fallback_nofma_squared_euclidean,
        xany = u8_xany_fallback_nofma_squared_euclidean,
    );

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = u16,
        register = Fallback,
        op = generic_dot_product,
        xconst = u16_xconst_fallback_nofma_dot,
        xany = u16_xany_fallback_nofma_dot,
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = u16,
        register = Fallback,
        op = generic_cosine,
        xconst = u16_xconst_fallback_nofma_cosine,
        xany = u16_xany_fallback_nofma_cosine,
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = u16,
        register = Fallback,
        op = generic_euclidean,
        xconst = u16_xconst_fallback_nofma_squared_euclidean,
        xany = u16_xany_fallback_nofma_squared_euclidean,
    );

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = u32,
        register = Fallback,
        op = generic_dot_product,
        xconst = u32_xconst_fallback_nofma_dot,
        xany = u32_xany_fallback_nofma_dot,
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = u32,
        register = Fallback,
        op = generic_cosine,
        xconst = u32_xconst_fallback_nofma_cosine,
        xany = u32_xany_fallback_nofma_cosine,
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = u32,
        register = Fallback,
        op = generic_euclidean,
        xconst = u32_xconst_fallback_nofma_squared_euclidean,
        xany = u32_xany_fallback_nofma_squared_euclidean,
    );

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = u64,
        register = Fallback,
        op = generic_dot_product,
        xconst = u64_xconst_fallback_nofma_dot,
        xany = u64_xany_fallback_nofma_dot,
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = u64,
        register = Fallback,
        op = generic_cosine,
        xconst = u64_xconst_fallback_nofma_cosine,
        xany = u64_xany_fallback_nofma_cosine,
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = u64,
        register = Fallback,
        op = generic_euclidean,
        xconst = u64_xconst_fallback_nofma_squared_euclidean,
        xany = u64_xany_fallback_nofma_squared_euclidean,
    );

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = f32,
        register = Fallback,
        op = generic_dot_product,
        xconst = f32_xconst_fallback_nofma_dot,
        xany = f32_xany_fallback_nofma_dot,
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = f32,
        register = Fallback,
        op = generic_cosine,
        xconst = f32_xconst_fallback_nofma_cosine,
        xany = f32_xany_fallback_nofma_cosine,
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = f32,
        register = Fallback,
        op = generic_euclidean,
        xconst = f32_xconst_fallback_nofma_squared_euclidean,
        xany = f32_xany_fallback_nofma_squared_euclidean,
    );

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = f64,
        register = Fallback,
        op = generic_dot_product,
        xconst = f64_xconst_fallback_nofma_dot,
        xany = f64_xany_fallback_nofma_dot,
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = f64,
        register = Fallback,
        op = generic_cosine,
        xconst = f64_xconst_fallback_nofma_cosine,
        xany = f64_xany_fallback_nofma_cosine,
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = f64,
        register = Fallback,
        op = generic_euclidean,
        xconst = f64_xconst_fallback_nofma_squared_euclidean,
        xany = f64_xany_fallback_nofma_squared_euclidean,
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
/// Vector space distance ops on the AVX2 supported architectures.
///
/// These methods **must** have AVX2 available to them at the time of calling
/// otherwise all of these methods become UB even if the input data is OK.
pub mod distance_ops_avx2 {
    use super::*;

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = i8,
        register = Avx2,
        op = generic_dot_product,
        xconst = i8_xconst_avx2_nofma_dot,
        xany = i8_xany_avx2_nofma_dot,
        features = "avx2"
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = i8,
        register = Avx2,
        op = generic_cosine,
        xconst = i8_xconst_avx2_nofma_cosine,
        xany = i8_xany_avx2_nofma_cosine,
        features = "avx2"
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = i8,
        register = Avx2,
        op = generic_euclidean,
        xconst = i8_xconst_avx2_nofma_squared_euclidean,
        xany = i8_xany_avx2_nofma_squared_euclidean,
        features = "avx2"
    );

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = i16,
        register = Avx2,
        op = generic_dot_product,
        xconst = i16_xconst_avx2_nofma_dot,
        xany = i16_xany_avx2_nofma_dot,
        features = "avx2"
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = i16,
        register = Avx2,
        op = generic_cosine,
        xconst = i16_xconst_avx2_nofma_cosine,
        xany = i16_xany_avx2_nofma_cosine,
        features = "avx2"
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = i16,
        register = Avx2,
        op = generic_euclidean,
        xconst = i16_xconst_avx2_nofma_squared_euclidean,
        xany = i16_xany_avx2_nofma_squared_euclidean,
        features = "avx2"
    );

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = i32,
        register = Avx2,
        op = generic_dot_product,
        xconst = i32_xconst_avx2_nofma_dot,
        xany = i32_xany_avx2_nofma_dot,
        features = "avx2"
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = i32,
        register = Avx2,
        op = generic_cosine,
        xconst = i32_xconst_avx2_nofma_cosine,
        xany = i32_xany_avx2_nofma_cosine,
        features = "avx2"
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = i32,
        register = Avx2,
        op = generic_euclidean,
        xconst = i32_xconst_avx2_nofma_squared_euclidean,
        xany = i32_xany_avx2_nofma_squared_euclidean,
        features = "avx2"
    );

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = i64,
        register = Avx2,
        op = generic_dot_product,
        xconst = i64_xconst_avx2_nofma_dot,
        xany = i64_xany_avx2_nofma_dot,
        features = "avx2"
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = i64,
        register = Avx2,
        op = generic_cosine,
        xconst = i64_xconst_avx2_nofma_cosine,
        xany = i64_xany_avx2_nofma_cosine,
        features = "avx2"
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = i64,
        register = Avx2,
        op = generic_euclidean,
        xconst = i64_xconst_avx2_nofma_squared_euclidean,
        xany = i64_xany_avx2_nofma_squared_euclidean,
        features = "avx2"
    );

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = u8,
        register = Avx2,
        op = generic_dot_product,
        xconst = u8_xconst_avx2_nofma_dot,
        xany = u8_xany_avx2_nofma_dot,
        features = "avx2"
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = u8,
        register = Avx2,
        op = generic_cosine,
        xconst = u8_xconst_avx2_nofma_cosine,
        xany = u8_xany_avx2_nofma_cosine,
        features = "avx2"
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = u8,
        register = Avx2,
        op = generic_euclidean,
        xconst = u8_xconst_avx2_nofma_squared_euclidean,
        xany = u8_xany_avx2_nofma_squared_euclidean,
        features = "avx2"
    );

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = u16,
        register = Avx2,
        op = generic_dot_product,
        xconst = u16_xconst_avx2_nofma_dot,
        xany = u16_xany_avx2_nofma_dot,
        features = "avx2"
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = u16,
        register = Avx2,
        op = generic_cosine,
        xconst = u16_xconst_avx2_nofma_cosine,
        xany = u16_xany_avx2_nofma_cosine,
        features = "avx2"
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = u16,
        register = Avx2,
        op = generic_euclidean,
        xconst = u16_xconst_avx2_nofma_squared_euclidean,
        xany = u16_xany_avx2_nofma_squared_euclidean,
        features = "avx2"
    );

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = u32,
        register = Avx2,
        op = generic_dot_product,
        xconst = u32_xconst_avx2_nofma_dot,
        xany = u32_xany_avx2_nofma_dot,
        features = "avx2"
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = u32,
        register = Avx2,
        op = generic_cosine,
        xconst = u32_xconst_avx2_nofma_cosine,
        xany = u32_xany_avx2_nofma_cosine,
        features = "avx2"
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = u32,
        register = Avx2,
        op = generic_euclidean,
        xconst = u32_xconst_avx2_nofma_squared_euclidean,
        xany = u32_xany_avx2_nofma_squared_euclidean,
        features = "avx2"
    );

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = u64,
        register = Avx2,
        op = generic_dot_product,
        xconst = u64_xconst_avx2_nofma_dot,
        xany = u64_xany_avx2_nofma_dot,
        features = "avx2"
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = u64,
        register = Avx2,
        op = generic_cosine,
        xconst = u64_xconst_avx2_nofma_cosine,
        xany = u64_xany_avx2_nofma_cosine,
        features = "avx2"
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = u64,
        register = Avx2,
        op = generic_euclidean,
        xconst = u64_xconst_avx2_nofma_squared_euclidean,
        xany = u64_xany_avx2_nofma_squared_euclidean,
        features = "avx2"
    );

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = f32,
        register = Avx2,
        op = generic_dot_product,
        xconst = f32_xconst_avx2_nofma_dot,
        xany = f32_xany_avx2_nofma_dot,
        features = "avx2"
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = f32,
        register = Avx2,
        op = generic_cosine,
        xconst = f32_xconst_avx2_nofma_cosine,
        xany = f32_xany_avx2_nofma_cosine,
        features = "avx2"
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = f32,
        register = Avx2,
        op = generic_euclidean,
        xconst = f32_xconst_avx2_nofma_squared_euclidean,
        xany = f32_xany_avx2_nofma_squared_euclidean,
        features = "avx2"
    );

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = f64,
        register = Avx2,
        op = generic_dot_product,
        xconst = f64_xconst_avx2_nofma_dot,
        xany = f64_xany_avx2_nofma_dot,
        features = "avx2"
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = f64,
        register = Avx2,
        op = generic_cosine,
        xconst = f64_xconst_avx2_nofma_cosine,
        xany = f64_xany_avx2_nofma_cosine,
        features = "avx2"
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = f64,
        register = Avx2,
        op = generic_euclidean,
        xconst = f64_xconst_avx2_nofma_squared_euclidean,
        xany = f64_xany_avx2_nofma_squared_euclidean,
        features = "avx2"
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
/// Vector space distance ops on the AVX2+FMA supported architectures.
///
/// These methods **must** have AVX2 + FMA available to them at the time of calling
/// otherwise all of these methods become UB even if the input data is OK.
pub mod distance_ops_avx2_fma {
    use super::*;

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = f32,
        register = Avx2Fma,
        op = generic_dot_product,
        xconst = f32_xconst_avx2_fma_dot,
        xany = f32_xany_avx2_fma_dot,
        features = "avx2",
        "fma"
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = f32,
        register = Avx2Fma,
        op = generic_cosine,
        xconst = f32_xconst_avx2_fma_cosine,
        xany = f32_xany_avx2_fma_cosine,
        features = "avx2",
        "fma"
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = f32,
        register = Avx2Fma,
        op = generic_euclidean,
        xconst = f32_xconst_avx2_fma_squared_euclidean,
        xany = f32_xany_avx2_fma_squared_euclidean,
        features = "avx2",
        "fma"
    );

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = f64,
        register = Avx2Fma,
        op = generic_dot_product,
        xconst = f64_xconst_avx2_fma_dot,
        xany = f64_xany_avx2_fma_dot,
        features = "avx2",
        "fma"
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = f64,
        register = Avx2Fma,
        op = generic_cosine,
        xconst = f64_xconst_avx2_fma_cosine,
        xany = f64_xany_avx2_fma_cosine,
        features = "avx2",
        "fma"
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = f64,
        register = Avx2Fma,
        op = generic_euclidean,
        xconst = f64_xconst_avx2_fma_squared_euclidean,
        xany = f64_xany_avx2_fma_squared_euclidean,
        features = "avx2",
        "fma"
    );
}

#[cfg(target_arch = "aarch64")]
/// Vector space distance ops on the AARCH64 NEON supported architectures.
///
/// These methods **must** have NEON available to them at the time of calling
/// otherwise all of these methods become UB even if the input data is OK.
///
/// The internal implementation of these operations current rely on auto-vectorization
/// by the compiler, which during tested it was able to perform well and beat out
/// some of my simple SIMD operations.
pub mod distance_ops_neon {
    use super::*;

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = i8,
        register = Fallback,
        op = generic_dot_product,
        xconst = i8_xconst_neon_nofma_dot,
        xany = i8_xany_neon_nofma_dot,
        features = "neon"
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = i8,
        register = Fallback,
        op = generic_cosine,
        xconst = i8_xconst_neon_nofma_cosine,
        xany = i8_xany_neon_nofma_cosine,
        features = "neon"
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = i8,
        register = Fallback,
        op = generic_euclidean,
        xconst = i8_xconst_neon_nofma_squared_euclidean,
        xany = i8_xany_neon_nofma_squared_euclidean,
        features = "neon"
    );

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = i16,
        register = Fallback,
        op = generic_dot_product,
        xconst = i16_xconst_neon_nofma_dot,
        xany = i16_xany_neon_nofma_dot,
        features = "neon"
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = i16,
        register = Fallback,
        op = generic_cosine,
        xconst = i16_xconst_neon_nofma_cosine,
        xany = i16_xany_neon_nofma_cosine,
        features = "neon"
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = i16,
        register = Fallback,
        op = generic_euclidean,
        xconst = i16_xconst_neon_nofma_squared_euclidean,
        xany = i16_xany_neon_nofma_squared_euclidean,
        features = "neon"
    );

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = i32,
        register = Fallback,
        op = generic_dot_product,
        xconst = i32_xconst_neon_nofma_dot,
        xany = i32_xany_neon_nofma_dot,
        features = "neon"
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = i32,
        register = Fallback,
        op = generic_cosine,
        xconst = i32_xconst_neon_nofma_cosine,
        xany = i32_xany_neon_nofma_cosine,
        features = "neon"
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = i32,
        register = Fallback,
        op = generic_euclidean,
        xconst = i32_xconst_neon_nofma_squared_euclidean,
        xany = i32_xany_neon_nofma_squared_euclidean,
        features = "neon"
    );

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = i64,
        register = Fallback,
        op = generic_dot_product,
        xconst = i64_xconst_neon_nofma_dot,
        xany = i64_xany_neon_nofma_dot,
        features = "neon"
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = i64,
        register = Fallback,
        op = generic_cosine,
        xconst = i64_xconst_neon_nofma_cosine,
        xany = i64_xany_neon_nofma_cosine,
        features = "neon"
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = i64,
        register = Fallback,
        op = generic_euclidean,
        xconst = i64_xconst_neon_nofma_squared_euclidean,
        xany = i64_xany_neon_nofma_squared_euclidean,
        features = "neon"
    );

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = u8,
        register = Fallback,
        op = generic_dot_product,
        xconst = u8_xconst_neon_nofma_dot,
        xany = u8_xany_neon_nofma_dot,
        features = "neon"
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = u8,
        register = Fallback,
        op = generic_cosine,
        xconst = u8_xconst_neon_nofma_cosine,
        xany = u8_xany_neon_nofma_cosine,
        features = "neon"
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = u8,
        register = Fallback,
        op = generic_euclidean,
        xconst = u8_xconst_neon_nofma_squared_euclidean,
        xany = u8_xany_neon_nofma_squared_euclidean,
        features = "neon"
    );

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = u16,
        register = Fallback,
        op = generic_dot_product,
        xconst = u16_xconst_neon_nofma_dot,
        xany = u16_xany_neon_nofma_dot,
        features = "neon"
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = u16,
        register = Fallback,
        op = generic_cosine,
        xconst = u16_xconst_neon_nofma_cosine,
        xany = u16_xany_neon_nofma_cosine,
        features = "neon"
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = u16,
        register = Fallback,
        op = generic_euclidean,
        xconst = u16_xconst_neon_nofma_squared_euclidean,
        xany = u16_xany_neon_nofma_squared_euclidean,
        features = "neon"
    );

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = u32,
        register = Fallback,
        op = generic_dot_product,
        xconst = u32_xconst_neon_nofma_dot,
        xany = u32_xany_neon_nofma_dot,
        features = "neon"
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = u32,
        register = Fallback,
        op = generic_cosine,
        xconst = u32_xconst_neonk_nofma_cosine,
        xany = u32_xany_neon_nofma_cosine,
        features = "neon"
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = u32,
        register = Fallback,
        op = generic_euclidean,
        xconst = u32_xconst_neon_nofma_squared_euclidean,
        xany = u32_xany_neon_nofma_squared_euclidean,
        features = "neon"
    );

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = u64,
        register = Fallback,
        op = generic_dot_product,
        xconst = u64_xconst_neon_nofma_dot,
        xany = u64_xany_neon_nofma_dot,
        features = "neon"
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = u64,
        register = Fallback,
        op = generic_cosine,
        xconst = u64_xconst_neon_nofma_cosine,
        xany = u64_xany_neon_nofma_cosine,
        features = "neon"
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = u64,
        register = Fallback,
        op = generic_euclidean,
        xconst = u64_xconst_neon_nofma_squared_euclidean,
        xany = u64_xany_neon_nofma_squared_euclidean,
        features = "neon"
    );

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = f32,
        register = Fallback,
        op = generic_dot_product,
        xconst = f32_xconst_neon_nofma_dot,
        xany = f32_xany_neon_nofma_dot,
        features = "neon"
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = f32,
        register = Fallback,
        op = generic_cosine,
        xconst = f32_xconst_neon_nofma_cosine,
        xany = f32_xany_neon_nofma_cosine,
        features = "neon"
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = f32,
        register = Fallback,
        op = generic_euclidean,
        xconst = f32_xconst_neon_nofma_squared_euclidean,
        xany = f32_xany_neon_nofma_squared_euclidean,
        features = "neon"
    );

    export_distance_op!(
        description = "Dot product of two vectors",
        ty = f64,
        register = Fallback,
        op = generic_dot_product,
        xconst = f64_xconst_neon_nofma_dot,
        xany = f64_xany_neon_nofma_dot,
        features = "neon"
    );
    export_distance_op!(
        description = "Cosine distance of two vectors",
        ty = f64,
        register = Fallback,
        op = generic_cosine,
        xconst = f64_xconst_neon_nofma_cosine,
        xany = f64_xany_neon_nofma_cosine,
        features = "neon"
    );
    export_distance_op!(
        description = "Squared Euclidean distance of two vectors",
        ty = f64,
        register = Fallback,
        op = generic_euclidean,
        xconst = f64_xconst_neon_nofma_squared_euclidean,
        xany = f64_xany_neon_nofma_squared_euclidean,
        features = "neon"
    );
}
