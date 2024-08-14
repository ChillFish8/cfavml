//! Miscellaneous aggregate operations
//!
//! These include routines that don't have a more suitable grouping (i.e. horizontal sum)
//! but still provide useful value having SIMD variants.

use crate::danger::{generic_sum, SimdRegister};
use crate::math::{AutoMath, Math};

macro_rules! define_sum_impl {
    (
        $name:ident,
        $imp:ident $(,)?
        $(target_features = $($feat:expr $(,)?)+)?
    ) => {
        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = r#"
            Performs a horizontal sum of all elements in vector `a` of size `dims` returning the total.

            ### Pseudocode
            ```py
            result = 0

            for i in range(dims):
                result += a[i]

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
            - The sizes of `a` must also be equal to size `dims` otherwise out of
              bounds access can occur.
        "#]
        pub unsafe fn $name<T>(
            dims: usize,
            a: &[T],
        ) -> T
        where
            T: Copy,
            crate::danger::$imp: SimdRegister<T>,
            AutoMath: Math<T>,
        {
            generic_sum::<T, crate::danger::$imp, AutoMath>(
                dims,
                a,
            )
        }
    };
}

define_sum_impl!(generic_fallback_sum, Fallback);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_sum_impl!(generic_avx2_sum, Avx2, target_features = "avx2");
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_sum_impl!(generic_avx512_sum, Avx512, target_features = "avx512f");
#[cfg(target_arch = "aarch64")]
define_sum_impl!(generic_neon_nofma_sum, Neon, target_features = "neon");
