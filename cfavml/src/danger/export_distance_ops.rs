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

macro_rules! define_cosine_impl {
    ($name:ident, $imp:ident $(,)? $(target_features = $($feat:expr $(,)?)+)?) => {
        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = r#"
            Calculates the cosine similarity distance between two vectors of size `dims`.

            ### Pseudocode

            ```ignore
            result = 0
            norm_a = 0
            norm_b = 0

            for i in range(dims):
                result += a[i] * b[i]
                norm_a += a[i] ** 2
                norm_b += b[i] ** 2

            if norm_a == 0.0 and norm_b == 0.0:
                return 0.0
            elif norm_a == 0.0 or norm_b == 0.0:
                return 1.0
            else:
                return 1.0 - (result / sqrt(norm_a * norm_b))
            ```

            # Safety

            This routine assumes:
        "#]
        $(

            #[doc = concat!("- ", $("**`+", $feat, "`** ", )*)]
            #[doc = "CPU features are available at runtime. Running on hardware _without_ this feature available will cause immediate UB."]
        )*
        #[doc = r#"
            - The sizes of `a` and `b` must also be equal to size `dims` otherwise out of
              bounds access can occur.
        "#]
        pub unsafe fn $name<T>(dims: usize, a: &[T], b: &[T]) -> T
        where
            T: Copy,
            crate::danger::$imp: SimdRegister<T>,
            AutoMath: Math<T>,
        {
            generic_cosine::<T, crate::danger::$imp, AutoMath>(
                dims,
                a,
                b,
            )
        }
    };
}

define_cosine_impl!(generic_fallback_cosine, Fallback);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_cosine_impl!(generic_avx2_cosine, Avx2, target_features = "avx2");
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_cosine_impl!(
    generic_avx2fma_cosine,
    Avx2Fma,
    target_features = "avx2",
    "fma"
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_cosine_impl!(generic_avx512_cosine, Avx512, target_features = "avx512f");
#[cfg(target_arch = "aarch64")]
define_cosine_impl!(generic_neon_cosine, Neon, target_features = "neon");

macro_rules! define_dot_impl {
    ($name:ident, $imp:ident $(,)? $(target_features = $($feat:expr $(,)?)+)?) => {
        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = r#"
            Calculates the dot product between two vectors of size `dims`.

            ### Pseudocode

            ```ignore
            result = 0;

            for i in range(dims):
                result += a[i] * b[i]

            return result
            ```

            # Safety

            This routine assumes:
        "#]
        $(

            #[doc = concat!("- ", $("**`+", $feat, "`** ", )*)]
            #[doc = "CPU features are available at runtime. Running on hardware _without_ this feature available will cause immediate UB."]
        )*
        #[doc = r#"
            - The sizes of `a` and `b` must also be equal to size `dims` otherwise out of
              bounds access can occur.
        "#]
        pub unsafe fn $name<T>(dims: usize, a: &[T], b: &[T]) -> T
        where
            T: Copy,
            crate::danger::$imp: SimdRegister<T>,
            AutoMath: Math<T>,
        {
            generic_dot::<T, crate::danger::$imp, AutoMath>(
                dims,
                a,
                b,
            )
        }
    };
}

define_dot_impl!(generic_fallback_dot, Fallback);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_dot_impl!(generic_avx2_dot, Avx2, target_features = "avx2");
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_dot_impl!(
    generic_avx2fma_dot,
    Avx2Fma,
    target_features = "avx2",
    "fma"
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_dot_impl!(generic_avx512_dot, Avx512, target_features = "avx512f");
#[cfg(target_arch = "aarch64")]
define_dot_impl!(generic_neon_dot, Neon, target_features = "neon");

macro_rules! define_euclidean_impl {
    ($name:ident, $imp:ident $(,)? $(target_features = $($feat:expr $(,)?)+)?) => {
        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = r#"
            Calculates the squared Euclidean distance between two vectors of size `dims`.

            ### Pseudocode

            ```ignore
            result = 0;

            for i in range(dims):
                diff = a[i] - b[i]
                result += diff ** 2

            return result
            ```

            # Safety

            This routine assumes:
        "#]
        $(

            #[doc = concat!("- ", $("**`+", $feat, "`** ", )*)]
            #[doc = "CPU features are available at runtime. Running on hardware _without_ this feature available will cause immediate UB."]
        )*
        #[doc = r#"
            - The sizes of `a` and `b` must also be equal to size `dims` otherwise out of
              bounds access can occur.
        "#]
        pub unsafe fn $name<T>(dims: usize, a: &[T], b: &[T]) -> T
        where
            T: Copy,
            crate::danger::$imp: SimdRegister<T>,
            AutoMath: Math<T>,
        {
            generic_squared_euclidean::<T, crate::danger::$imp, AutoMath>(
                dims,
                a,
                b,
            )
        }
    };
}

define_euclidean_impl!(generic_fallback_squared_euclidean, Fallback);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_euclidean_impl!(
    generic_avx2_squared_euclidean,
    Avx2,
    target_features = "avx2"
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_euclidean_impl!(
    generic_avx2fma_squared_euclidean,
    Avx2Fma,
    target_features = "avx2",
    "fma"
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_euclidean_impl!(
    generic_avx512_squared_euclidean,
    Avx512,
    target_features = "avx512f"
);
#[cfg(target_arch = "aarch64")]
define_euclidean_impl!(
    generic_neon_squared_euclidean,
    Neon,
    target_features = "neon"
);

macro_rules! define_norm_impl {
    ($name:ident, $imp:ident $(,)? $(target_features = $($feat:expr $(,)?)+)?) => {
        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = r#"
            Calculates the squared L2 norm of vector `a` of size `dims`.

            ### Pseudocode

            ```ignore
            result = 0;

            for i in range(dims):
                result += a[i] ** 2

            return result
            ```

            # Safety

            This routine assumes:
        "#]
        $(

            #[doc = concat!("- ", $("**`+", $feat, "`** ", )*)]
            #[doc = "CPU features are available at runtime. Running on hardware _without_ this feature available will cause immediate UB."]
        )*
        #[doc = r#"
            - The sizes of `a` and `b` must also be equal to size `dims` otherwise out of
              bounds access can occur.
        "#]
        pub unsafe fn $name<T>(dims: usize, a: &[T]) -> T
        where
            T: Copy,
            crate::danger::$imp: SimdRegister<T>,
            AutoMath: Math<T>,
        {
            generic_squared_norm::<T, crate::danger::$imp, AutoMath>(
                dims,
                a,
            )
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
    target_features = "avx512f"
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

                        let actual = unsafe { [< $variant _cosine >](l1.len(), &l1, &l2) };
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

                        let actual = unsafe { [< $variant _dot >](l1.len(), &l1, &l2) };
                        let expected: $t = crate::test_utils::simple_dot(&l1, &l2);
                        assert!(
                            AutoMath::is_close(actual, expected),
                            "Routine result does not match expected, {actual:?} vs {expected:?}",
                        );
                    }

                    #[test]
                    fn [< $variant _euclidean_ $t >]() {
                        let (l1, l2) = crate::test_utils::get_sample_vectors::<$t>(533);

                        let actual = unsafe { [< $variant _squared_euclidean >](l1.len(), &l1, &l2) };
                        let expected: $t = crate::test_utils::simple_euclidean(&l1, &l2);
                        assert!(
                            AutoMath::is_close(actual, expected),
                            "Routine result does not match expected, {actual:?} vs {expected:?}",
                        );
                    }

                    #[test]
                    fn [< $variant _norm_ $t >]() {
                        let (l1, _) = crate::test_utils::get_sample_vectors::<$t>(533);

                        let actual = unsafe { [< $variant _squared_norm >](l1.len(), &l1) };
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
