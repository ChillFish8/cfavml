//! Common arithmetic operations
//!
//! I.e. Add, Sub, Mul, Div...

use crate::buffer::WriteOnlyBuffer;
use crate::danger::core_simd_api::Hypot;
use crate::danger::{generic_hypot_vertical, SimdRegister};
use crate::math::{AutoMath, Math, Numeric};
use crate::mem_loader::{IntoMemLoader, MemLoader};

macro_rules! define_arithmetic_impls {
    (
        $hypot_name:ident,
        $imp:ident $(,)?
        $(target_features = $($feat:expr $(,)?)+)?
    ) => {
        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = include_str!("../export_docs/arithmetic_hypot_vertical.md")]
        $(

            #[doc = concat!("- ", $("**`+", $feat, "`** ", )*)]
            #[doc = "CPU features are available at runtime. Running on hardware _without_ this feature available will cause immediate UB."]
        )*
        pub unsafe fn $hypot_name<T, B1, B2, B3>(
            a: B1,
            b: B2,
            result: &mut [B3],
        )
        where
            T: Copy,
            B1: IntoMemLoader<T>,
            B1::Loader: MemLoader<Value = T>,
            B2: IntoMemLoader<T>,
            B2::Loader: MemLoader<Value = T>,
            crate::danger::$imp: SimdRegister<T>+ Hypot<T>,
            AutoMath: Math<T> + Numeric<T>,
            for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = T>,
        {
            generic_hypot_vertical::<T, crate::danger::$imp, AutoMath, B1, B2, B3>(
                a,
                b,
                result,
            )
        }
    };
}

define_arithmetic_impls!(generic_fallback_hypot_vertical, Fallback,);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_arithmetic_impls!(generic_avx2_hypot_vertical, Avx2, target_features = "avx2");
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_arithmetic_impls!(
    generic_avx512_hypot_vertical,
    Avx512,
    target_features = "avx512f",
    "avx512bw"
);
#[cfg(target_arch = "aarch64")]
define_arithmetic_impls!(generic_neon_hypot_vertical, Neon, target_features = "neon");

#[cfg(test)]
mod tests {
    use num_traits::{Float, FloatConst};

    use super::*;

    macro_rules! define_inner_test {
        ($variant:ident, op = $op:ident, ty = $t:ident) => {
            paste::paste! {
                #[test]
                fn [< $variant _ $op _value_ $t >]() {
                    let (l1, _) = crate::test_utils::get_sample_vectors::<$t>(533);

                    let mut result = vec![$t::default(); 533];
                    unsafe { [< $variant _ $op _vertical >](&l1, 2 as $t, &mut result) };

                    let expected = l1.iter()
                        .copied()
                        .map(|v| AutoMath::$op(v, 2 as $t))
                        .collect::<Vec<_>>();

                    for ((initial, expected), actual) in l1.iter().zip(&expected).zip(&result) {
                        let ulps_diff = get_diff_ulps(*expected, *actual);
                        assert!(
                            ulps_diff.abs() <= 1,
                            "result differs by more than 1 ULP:\n initial inputs: {}, 2\n  expected: {} actual: {}\nulps diff: {}",
                            initial, expected, actual, ulps_diff
                        );

                    }

                }

                #[test]
                fn [< $variant _ $op _vector_ $t >]() {
                    let (l1, l2) = crate::test_utils::get_sample_vectors::<$t>(533);

                    let mut result = vec![$t::default(); 533];
                    unsafe { [< $variant _ $op _vertical >](&l1, &l2, &mut result) };

                    let expected = l1.iter()
                        .copied()
                        .zip(l2.iter().copied())
                        .map(|(a, b)| AutoMath::$op(a, b))
                        .collect::<Vec<_>>();
                    for ((initial, expected), actual) in l1.iter().zip(&expected).zip(&result) {
                        let ulps_diff = get_diff_ulps(*expected, *actual);
                        assert!(
                            ulps_diff.abs() <= 1,
                            "result differs by more than 1 ULP:\n initial inputs: {}, 2\n  expected: {} actual: {}\nulps diff: {}",
                            initial, expected, actual, ulps_diff
                        );
                    }
                }
            }
        };
    }

    fn get_diff_ulps<T>(a: T, b: T) -> i64
    where
        T: Float + FloatConst,
    {
        let (a_mant, a_exp, a_sign) = a.integer_decode();
        let (b_mant, b_exp, b_sign) = b.integer_decode();
        assert!(a_sign == b_sign);
        assert!(a_exp == b_exp);
        a_mant as i64 - b_mant as i64
    }

    macro_rules! define_numeric_test {
        ($variant:ident, types = $($t:ident $(,)?)+) => {
            $(
                define_inner_test!($variant, op = hypot, ty = $t);

            )*
        };
    }

    define_numeric_test!(generic_fallback, types = f32, f64,);
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    ))]
    define_numeric_test!(generic_avx2, types = f32, f64,);
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "nightly",
        target_feature = "avx512f"
    ))]
    define_numeric_test!(generic_avx512, types = f32, f64,);
    #[cfg(target_arch = "aarch64")]
    define_numeric_test!(generic_neon, types = f32, f64,);
}
