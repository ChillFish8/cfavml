//! Safe but somewhat low-level variants of the distance operations in CFAVML.
//!
//! In general, I would recommend using the higher level generic functions api which provides
//! some syntax sugar over these traits.

use crate::danger::export_distance_ops;
use crate::math::{AutoMath, Math};

/// Various spacial distance operations between vectors.
pub trait DistanceOps: Sized + Copy
where
    AutoMath: Math<Self>
{
    /// Calculates the cosine similarity distance of vectors `a` and `b` of size `dims`.
    ///
    /// ### Pseudocode
    ///
    /// ```ignore
    /// result = 0
    /// norm_a = 0
    /// norm_b = 0
    ///
    /// for i in range(dims):
    ///     result += a[i] * b[i]
    ///     norm_a += a[i] ** 2
    ///     norm_b += b[i] ** 2
    ///
    /// if norm_a == 0.0 and norm_b == 0.0:
    ///     return 0.0
    /// elif norm_a == 0.0 or norm_b == 0.0:
    ///     return 1.0
    /// else:
    ///     return 1.0 - (result / sqrt(norm_a * norm_b))
    /// ```
    ///
    /// ### Panics
    ///
    /// This function will panic if vectors `a` and `b` do not match size `dims`.
    fn cosine(dims: usize, a: &[Self], b: &[Self]) -> Self;

    /// Calculates the cosine similarity distance of vectors `a` and `b` of size `dims`.
    ///
    /// ### Pseudocode
    ///
    /// ```ignore
    /// result = 0
    ///
    /// for i in range(dims):
    ///     result += a[i] * b[i]
    ///
    /// return result
    /// ```
    ///
    /// ### Panics
    ///
    /// This function will panic if vectors `a` and `b` do not match size `dims`.
    fn dot(dims: usize, a: &[Self], b: &[Self]) -> Self;

    /// Calculates the squared Euclidean distance of vectors `a` and `b` of size `dims`.
    ///
    /// ### Pseudocode
    ///
    /// ```ignore
    /// result = 0
    ///
    /// for i in range(dims):
    ///     diff = a[i] - b[i]
    ///     result += diff * diff
    ///
    /// return result
    /// ```
    ///
    /// ### Panics
    ///
    /// This function will panic if vectors `a` and `b` do not match size `dims`.
    fn squared_euclidean(dims: usize, a: &[Self], b: &[Self]) -> Self;

    /// Calculates the squared L2 norm of vector `a` of size `dims`.
    ///
    /// ### Pseudocode
    ///
    /// ```ignore
    /// result = 0
    ///
    /// for i in range(dims):
    ///     result += a[i] * a[i]
    ///
    /// return result
    /// ```
    ///
    /// ### Panics
    ///
    /// This function will panic if vectors `a` does not match size `dims`.
    fn squared_norm(dims: usize, a: &[Self]) -> Self;
}

macro_rules! float_distance_ops {
    ($t:ty) => {
        impl DistanceOps for $t {
            fn cosine(dims: usize, a: &[Self], b: &[Self]) -> Self {
                assert_eq!(a.len(), dims, "Input vector `a` does not match size `dims`");
                assert_eq!(b.len(), dims, "Input vector `b` does not match size `dims`");

                unsafe {
                    crate::dispatch!(
                        avx512 = export_distance_ops::generic_avx512_cosine,
                        avx2fma = export_distance_ops::generic_avx2fma_cosine,
                        avx2 = export_distance_ops::generic_avx2_cosine,
                        neon = export_distance_ops::generic_neon_cosine,
                        fallback = export_distance_ops::generic_fallback_cosine,
                        args = (dims, a, b)
                    )
                }
            }

            fn dot(dims: usize, a: &[Self], b: &[Self]) -> Self {
                assert_eq!(a.len(), dims, "Input vector `a` does not match size `dims`");
                assert_eq!(b.len(), dims, "Input vector `b` does not match size `dims`");

                unsafe {
                    crate::dispatch!(
                        avx512 = export_distance_ops::generic_avx512_dot,
                        avx2fma = export_distance_ops::generic_avx2fma_dot,
                        avx2 = export_distance_ops::generic_avx2_dot,
                        neon = export_distance_ops::generic_neon_dot,
                        fallback = export_distance_ops::generic_fallback_dot,
                        args = (dims, a, b)
                    )
                }
            }

            fn squared_euclidean(dims: usize, a: &[Self], b: &[Self]) -> Self {
                assert_eq!(a.len(), dims, "Input vector `a` does not match size `dims`");
                assert_eq!(b.len(), dims, "Input vector `b` does not match size `dims`");

                unsafe {
                    crate::dispatch!(
                        avx512 = export_distance_ops::generic_avx512_squared_euclidean,
                        avx2fma = export_distance_ops::generic_avx2fma_squared_euclidean,
                        avx2 = export_distance_ops::generic_avx2_squared_euclidean,
                        neon = export_distance_ops::generic_neon_squared_euclidean,
                        fallback =
                            export_distance_ops::generic_fallback_squared_euclidean,
                        args = (dims, a, b)
                    )
                }
            }

            fn squared_norm(dims: usize, a: &[Self]) -> Self {
                assert_eq!(a.len(), dims, "Input vector `a` does not match size `dims`");

                unsafe {
                    crate::dispatch!(
                        avx512 = export_distance_ops::generic_avx512_squared_norm,
                        avx2fma = export_distance_ops::generic_avx2fma_squared_norm,
                        avx2 = export_distance_ops::generic_avx2_squared_norm,
                        neon = export_distance_ops::generic_neon_squared_norm,
                        fallback = export_distance_ops::generic_fallback_squared_norm,
                        args = (dims, a)
                    )
                }
            }
        }
    };
}

macro_rules! scalar_distance_ops {
    ($t:ty) => {
        impl DistanceOps for $t {
            fn cosine(dims: usize, a: &[Self], b: &[Self]) -> Self {
                assert_eq!(a.len(), dims, "Input vector `a` does not match size `dims`");
                assert_eq!(b.len(), dims, "Input vector `b` does not match size `dims`");

                unsafe {
                    crate::dispatch!(
                        avx512 = export_distance_ops::generic_avx512_cosine,
                        avx2 = export_distance_ops::generic_avx2_cosine,
                        neon = export_distance_ops::generic_neon_cosine,
                        fallback = export_distance_ops::generic_fallback_cosine,
                        args = (dims, a, b)
                    )
                }
            }

            fn dot(dims: usize, a: &[Self], b: &[Self]) -> Self {
                assert_eq!(a.len(), dims, "Input vector `a` does not match size `dims`");
                assert_eq!(b.len(), dims, "Input vector `b` does not match size `dims`");

                unsafe {
                    crate::dispatch!(
                        avx512 = export_distance_ops::generic_avx512_dot,
                        avx2 = export_distance_ops::generic_avx2_dot,
                        neon = export_distance_ops::generic_neon_dot,
                        fallback = export_distance_ops::generic_fallback_dot,
                        args = (dims, a, b)
                    )
                }
            }

            fn squared_euclidean(dims: usize, a: &[Self], b: &[Self]) -> Self {
                assert_eq!(a.len(), dims, "Input vector `a` does not match size `dims`");
                assert_eq!(b.len(), dims, "Input vector `b` does not match size `dims`");

                unsafe {
                    crate::dispatch!(
                        avx512 = export_distance_ops::generic_avx512_squared_euclidean,
                        avx2 = export_distance_ops::generic_avx2_squared_euclidean,
                        neon = export_distance_ops::generic_neon_squared_euclidean,
                        fallback =
                            export_distance_ops::generic_fallback_squared_euclidean,
                        args = (dims, a, b)
                    )
                }
            }

            fn squared_norm(dims: usize, a: &[Self]) -> Self {
                assert_eq!(a.len(), dims, "Input vector `a` does not match size `dims`");

                unsafe {
                    crate::dispatch!(
                        avx512 = export_distance_ops::generic_avx512_squared_norm,
                        avx2 = export_distance_ops::generic_avx2_squared_norm,
                        neon = export_distance_ops::generic_neon_squared_norm,
                        fallback = export_distance_ops::generic_fallback_squared_norm,
                        args = (dims, a)
                    )
                }
            }
        }
    };
}

float_distance_ops!(f32);
float_distance_ops!(f64);
scalar_distance_ops!(i8);
scalar_distance_ops!(i16);
scalar_distance_ops!(i32);
scalar_distance_ops!(i64);
scalar_distance_ops!(u8);
scalar_distance_ops!(u16);
scalar_distance_ops!(u32);
scalar_distance_ops!(u64);
