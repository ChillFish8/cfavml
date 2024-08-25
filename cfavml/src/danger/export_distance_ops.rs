//! Operations typically used for spacial distance calculations
//!
//! These operations are well suited for vector search situations, although things like
//! dot product are more generic than simply vector search.

use crate::danger::{
    generic_cosine,
    generic_dot,
    generic_squared_euclidean,
    generic_squared_norm,
    SimdRegister,
};
use crate::math::{AutoMath, Math};
use crate::mem_loader::{IntoMemLoader, MemLoader};

macro_rules! define_dist_impl {
    (
        name = $name:ident,
        op = $op:ident,
        doc = $doc:expr,
        $imp:ident $(,)?
        $(target_features = $($feat:expr $(,)?)+)?
    ) => {
        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = include_str!($doc)]
        $(

            #[doc = concat!("- ", $("**`+", $feat, "`** ", )*)]
            #[doc = "CPU features are available at runtime. Running on hardware _without_ this feature available will cause immediate UB."]
        )*
        pub unsafe fn $name<T, B1, B2>(a: B1, b: B2) -> T
        where
            T: Copy,
            B1: IntoMemLoader<T>,
            B1::Loader: MemLoader<Value = T>,
            B2: IntoMemLoader<T>,
            B2::Loader: MemLoader<Value = T>,
            crate::danger::$imp: SimdRegister<T>,
            AutoMath: Math<T>,
        {
            $op::<T, crate::danger::$imp, AutoMath, _, _>(
                a,
                b,
            )
        }
    };
}

define_dist_impl!(
    name = generic_fallback_cosine,
    op = generic_cosine,
    doc = "../export_docs/dist_cosine.md",
    Fallback,
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_dist_impl!(
    name = generic_avx2_cosine,
    op = generic_cosine,
    doc = "../export_docs/dist_cosine.md",
    Avx2,
    target_features = "avx2",
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_dist_impl!(
    name = generic_avx2fma_cosine,
    op = generic_cosine,
    doc = "../export_docs/dist_cosine.md",
    Avx2Fma,
    target_features = "avx2",
    "fma"
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_dist_impl!(
    name = generic_avx512_cosine,
    op = generic_cosine,
    doc = "../export_docs/dist_cosine.md",
    Avx512,
    target_features = "avx512f",
    "avx512bw"
);
#[cfg(target_arch = "aarch64")]
define_dist_impl!(
    name = generic_neon_cosine,
    op = generic_cosine,
    doc = "../export_docs/dist_cosine.md",
    Neon,
    target_features = "neon",
);

define_dist_impl!(
    name = generic_fallback_dot,
    op = generic_dot,
    doc = "../export_docs/dist_dot.md",
    Fallback,
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_dist_impl!(
    name = generic_avx2_dot,
    op = generic_dot,
    doc = "../export_docs/dist_dot.md",
    Avx2,
    target_features = "avx2"
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_dist_impl!(
    name = generic_avx2fma_dot,
    op = generic_dot,
    doc = "../export_docs/dist_dot.md",
    Avx2Fma,
    target_features = "avx2",
    "fma"
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_dist_impl!(
    name = generic_avx512_dot,
    op = generic_dot,
    doc = "../export_docs/dist_dot.md",
    Avx512,
    target_features = "avx512f",
    "avx512bw"
);
#[cfg(target_arch = "aarch64")]
define_dist_impl!(
    name = generic_neon_dot,
    op = generic_dot,
    doc = "../export_docs/dist_dot.md",
    Neon,
    target_features = "neon"
);

define_dist_impl!(
    name = generic_fallback_squared_euclidean,
    op = generic_squared_euclidean,
    doc = "../export_docs/dist_euclidean.md",
    Fallback,
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_dist_impl!(
    name = generic_avx2_squared_euclidean,
    op = generic_squared_euclidean,
    doc = "../export_docs/dist_euclidean.md",
    Avx2,
    target_features = "avx2"
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_dist_impl!(
    name = generic_avx2fma_squared_euclidean,
    op = generic_squared_euclidean,
    doc = "../export_docs/dist_euclidean.md",
    Avx2Fma,
    target_features = "avx2",
    "fma"
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_dist_impl!(
    name = generic_avx512_squared_euclidean,
    op = generic_squared_euclidean,
    doc = "../export_docs/dist_euclidean.md",
    Avx512,
    target_features = "avx512f",
    "avx512bw"
);
#[cfg(target_arch = "aarch64")]
define_dist_impl!(
    name = generic_neon_squared_euclidean,
    op = generic_squared_euclidean,
    doc = "../export_docs/dist_euclidean.md",
    Neon,
    target_features = "neon"
);

macro_rules! define_norm_impl {
    ($name:ident, $imp:ident $(,)? $(target_features = $($feat:expr $(,)?)+)?) => {
        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = include_str!("../export_docs/dist_norm.md")]
        $(

            #[doc = concat!("- ", $("**`+", $feat, "`** ", )*)]
            #[doc = "CPU features are available at runtime. Running on hardware _without_ this feature available will cause immediate UB."]
        )*
        pub unsafe fn $name<T, B1>(a: B1) -> T
        where
            T: Copy,
            B1: IntoMemLoader<T>,
            B1::Loader: MemLoader<Value = T>,
            crate::danger::$imp: SimdRegister<T>,
            AutoMath: Math<T>,
        {
            generic_squared_norm::<T, crate::danger::$imp, AutoMath, _>(a)
        }
    };
}

define_norm_impl!(generic_fallback_squared_norm, Fallback);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_norm_impl!(generic_avx2_squared_norm, Avx2, target_features = "avx2");
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_norm_impl!(
    generic_avx2fma_squared_norm,
    Avx2Fma,
    target_features = "avx2",
    "fma",
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_norm_impl!(
    generic_avx512_squared_norm,
    Avx512,
    target_features = "avx512f",
    "avx512bw"
);
#[cfg(target_arch = "aarch64")]
define_norm_impl!(generic_neon_squared_norm, Neon, target_features = "neon");

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! define_cosine_extra_test {
        ($variant:ident, types = $($t:ident $(,)?)+) => {
            $(
                paste::paste! {
                    #[test]
                    fn [< $variant _cosine_ $t >]() {
                        let (l1, l2) = crate::test_utils::get_sample_vectors::<$t>(533);

                        let actual = unsafe { [< $variant _cosine >](&l1, &l2) };
                        let expected: $t = crate::test_utils::simple_cosine(&l1, &l2);
                        assert!(
                            AutoMath::is_close(actual, expected),
                            "Routine result does not match expected, {actual:?} vs {expected:?}",
                        );
                    }

                }
            )*
        };
    }

    macro_rules! define_distance_test {
        ($variant:ident, types = $($t:ident $(,)?)+) => {
            $(
                paste::paste! {
                    #[test]
                    fn [< $variant _dot_ $t >]() {
                        let (l1, l2) = crate::test_utils::get_sample_vectors::<$t>(533);

                        let actual = unsafe { [< $variant _dot >](&l1, &l2) };
                        let expected: $t = crate::test_utils::simple_dot(&l1, &l2);
                        assert!(
                            AutoMath::is_close(actual, expected),
                            "Routine result does not match expected, {actual:?} vs {expected:?}",
                        );
                    }

                    #[test]
                    fn [< $variant _euclidean_ $t >]() {
                        let (l1, l2) = crate::test_utils::get_sample_vectors::<$t>(533);

                        let actual = unsafe { [< $variant _squared_euclidean >](&l1, &l2) };
                        let expected: $t = crate::test_utils::simple_euclidean(&l1, &l2);
                        assert!(
                            AutoMath::is_close(actual, expected),
                            "Routine result does not match expected, {actual:?} vs {expected:?}",
                        );
                    }

                    #[test]
                    fn [< $variant _norm_ $t >]() {
                        let (l1, _) = crate::test_utils::get_sample_vectors::<$t>(533);

                        let actual = unsafe { [< $variant _squared_norm >](&l1) };
                        let expected: $t = crate::test_utils::simple_dot(&l1, &l1);
                        assert!(
                            AutoMath::is_close(actual, expected),
                            "Routine result does not match expected, {actual:?} vs {expected:?}",
                        );
                    }
                }
            )*
        };
    }

    define_distance_test!(
        generic_fallback,
        types = f32,
        f64,
        i8,
        i16,
        i32,
        i64,
        u8,
        u16,
        u32,
        u64
    );
    define_cosine_extra_test!(generic_fallback, types = f32, f64, i8, u8);

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    ))]
    define_distance_test!(
        generic_avx2,
        types = f32,
        f64,
        i8,
        i16,
        i32,
        i64,
        u8,
        u16,
        u32,
        u64
    );
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    ))]
    define_cosine_extra_test!(generic_avx2, types = f32, f64, i8, u8);

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    define_distance_test!(generic_avx2fma, types = f32, f64);
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    define_cosine_extra_test!(generic_avx2fma, types = f32, f64);

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "nightly",
        target_feature = "avx512f"
    ))]
    define_distance_test!(
        generic_avx512,
        types = f32,
        f64,
        i8,
        i16,
        i32,
        i64,
        u8,
        u16,
        u32,
        u64
    );
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "nightly",
        target_feature = "avx512f"
    ))]
    define_cosine_extra_test!(generic_avx512, types = f32, f64, i8, u8);

    #[cfg(target_arch = "aarch64")]
    define_distance_test!(
        generic_neon,
        types = f32,
        f64,
        i8,
        i16,
        i32,
        i64,
        u8,
        u16,
        u32,
        u64
    );
    #[cfg(target_arch = "aarch64")]
    define_cosine_extra_test!(generic_neon, types = f32, f64, i8, u8);
}
