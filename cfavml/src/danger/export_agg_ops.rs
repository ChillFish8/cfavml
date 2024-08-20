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
        #[doc = include_str!("../export_docs/agg_horizontal_sum.md")]
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
define_sum_impl!(
    generic_avx512_sum,
    Avx512,
    target_features = "avx512f",
    "avx512bw"
);
#[cfg(target_arch = "aarch64")]
define_sum_impl!(generic_neon_sum, Neon, target_features = "neon");

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! define_agg_test {
        ($variant:ident, types = $($t:ident $(,)?)+) => {
            $(
                paste::paste! {
                    #[test]
                    fn [< $variant _sum_ $t >]() {
                        let (l1, _) = crate::test_utils::get_sample_vectors::<$t>(533);

                        let actual_sum = unsafe { [< $variant _sum >](l1.len(), &l1) };
                        let expected_sum: $t = l1.iter().fold($t::default(), |a, b| AutoMath::add(a, *b));
                        assert!(
                            AutoMath::is_close(actual_sum, expected_sum),
                            "Routine result does not match expected sum, {actual_sum:?} vs {expected_sum:?}",
                        );
                    }
                }
            )*
        };
    }

    define_agg_test!(
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
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    ))]
    define_agg_test!(
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
        feature = "nightly",
        target_feature = "avx512f"
    ))]
    define_agg_test!(
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
    #[cfg(target_arch = "aarch64")]
    define_agg_test!(
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
}
