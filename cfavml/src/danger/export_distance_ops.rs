//! Operations typically used for spacial distance calculations
//!
//! These operations are well suited for vector search situations, although things like
//! dot product are more generic than simply vector search.

use crate::danger::{
    generic_cosine,
    generic_dot_product,
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
            ```py
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

define_cosine_impl!(generic_fallback_nofma_cosine, Fallback);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_cosine_impl!(generic_avx2_nofma_cosine, Avx2, target_features = "avx2");
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_cosine_impl!(
    generic_avx2_fma_cosine,
    Avx2Fma,
    target_features = "avx2",
    "fma"
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_cosine_impl!(
    generic_avx512_fma_cosine,
    Avx512,
    target_features = "avx512f"
);
#[cfg(target_arch = "aarch64")]
define_cosine_impl!(generic_neon_fma_cosine, Neon, target_features = "neon");

macro_rules! define_dot_impl {
    ($name:ident, $imp:ident $(,)? $(target_features = $($feat:expr $(,)?)+)?) => {
        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = r#"
            Calculates the dot product between two vectors of size `dims`.

            ### Pseudocode
            ```py
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
            generic_dot_product::<T, crate::danger::$imp, AutoMath>(
                dims,
                a,
                b,
            )
        }
    };
}

define_dot_impl!(generic_fallback_nofma_dot_product, Fallback);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_dot_impl!(
    generic_avx2_nofma_dot_product,
    Avx2,
    target_features = "avx2"
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_dot_impl!(
    generic_avx2_fma_dot_product,
    Avx2Fma,
    target_features = "avx2",
    "fma"
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_dot_impl!(
    generic_avx512_fma_dot_product,
    Avx512,
    target_features = "avx512f"
);
#[cfg(target_arch = "aarch64")]
define_dot_impl!(generic_neon_fma_dot_product, Neon, target_features = "neon");

macro_rules! define_euclidean_impl {
    ($name:ident, $imp:ident $(,)? $(target_features = $($feat:expr $(,)?)+)?) => {
        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = r#"
            Calculates the squared Euclidean distance between two vectors of size `dims`.

            ### Pseudocode
            ```py
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

define_euclidean_impl!(generic_fallback_nofma_squared_euclidean, Fallback);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_euclidean_impl!(
    generic_avx2_nofma_squared_euclidean,
    Avx2,
    target_features = "avx2"
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_euclidean_impl!(
    generic_avx2_fma_squared_euclidean,
    Avx2Fma,
    target_features = "avx2",
    "fma"
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_euclidean_impl!(
    generic_avx512_fma_squared_euclidean,
    Avx512,
    target_features = "avx512f"
);
#[cfg(target_arch = "aarch64")]
define_euclidean_impl!(
    generic_neon_fma_squared_euclidean,
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
            ```py
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

define_norm_impl!(generic_fallback_nofma_squared_norm, Fallback);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_norm_impl!(
    generic_avx2_nofma_squared_norm,
    Avx2,
    target_features = "avx2"
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_norm_impl!(
    generic_avx2_fma_squared_norm,
    Avx2Fma,
    target_features = "avx2",
    "fma",
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_norm_impl!(
    generic_avx512_fma_squared_norm,
    Avx512,
    target_features = "avx512f"
);
#[cfg(target_arch = "aarch64")]
define_norm_impl!(
    generic_neon_fma_squared_norm,
    Neon,
    target_features = "neon"
);
