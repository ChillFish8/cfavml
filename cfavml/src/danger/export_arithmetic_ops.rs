//! Common arithmetic operations
//!
//! I.e. Add, Sub, Mul, Div...

use crate::buffer::WriteOnlyBuffer;
use crate::danger::{
    generic_add_vertical,
    generic_div_vertical,
    generic_mul_vertical,
    generic_sub_vertical,
    SimdRegister,
};
use crate::math::{AutoMath, Math};
use crate::mem_loader::{IntoMemLoader, MemLoader};

macro_rules! define_arithmetic_impls {
    (
        add = $add_name:ident,
        sub = $sub_name:ident,
        mul = $mul_name:ident,
        div = $div_name:ident,
        $imp:ident $(,)?
        $(target_features = $($feat:expr $(,)?)+)?
    ) => {
        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = include_str!("../export_docs/arithmetic_add_vertical.md")]
        $(

            #[doc = concat!("- ", $("**`+", $feat, "`** ", )*)]
            #[doc = "CPU features are available at runtime. Running on hardware _without_ this feature available will cause immediate UB."]
        )*
        pub unsafe fn $add_name<T, B1, B2, B3>(
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
            crate::danger::$imp: SimdRegister<T>,
            AutoMath: Math<T>,
            for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = T>,
        {
            generic_add_vertical::<T, crate::danger::$imp, AutoMath, B1, B2, B3>(
                a,
                b,
                result,
            )
        }

        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = include_str!("../export_docs/arithmetic_sub_vertical.md")]
        $(

            #[doc = concat!("- ", $("**`+", $feat, "`** ", )*)]
            #[doc = "CPU features are available at runtime. Running on hardware _without_ this feature available will cause immediate UB."]
        )*
        pub unsafe fn $sub_name<T, B1, B2, B3>(
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
            crate::danger::$imp: SimdRegister<T>,
            AutoMath: Math<T>,
            for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = T>,
        {
            generic_sub_vertical::<T, crate::danger::$imp, AutoMath, B1, B2, B3>(
                a,
                b,
                result,
            )
        }

        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = include_str!("../export_docs/arithmetic_mul_vertical.md")]
        $(

            #[doc = concat!("- ", $("**`+", $feat, "`** ", )*)]
            #[doc = "CPU features are available at runtime. Running on hardware _without_ this feature available will cause immediate UB."]
        )*
        pub unsafe fn $mul_name<T, B1, B2, B3>(
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
            crate::danger::$imp: SimdRegister<T>,
            AutoMath: Math<T>,
            for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = T>,
        {
            generic_mul_vertical::<T, crate::danger::$imp, AutoMath, B1, B2, B3>(
                a,
                b,
                result,
            )
        }

        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = include_str!("../export_docs/arithmetic_div_vertical.md")]
        $(

            #[doc = concat!("- ", $("**`+", $feat, "`** ", )*)]
            #[doc = "CPU features are available at runtime. Running on hardware _without_ this feature available will cause immediate UB."]
        )*
        pub unsafe fn $div_name<T, B1, B2, B3>(
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
            crate::danger::$imp: SimdRegister<T>,
            AutoMath: Math<T>,
            for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = T>,
        {
            generic_div_vertical::<T, crate::danger::$imp, AutoMath, B1, B2, B3>(
                a,
                b,
                result,
            )
        }
    };
}

define_arithmetic_impls!(
    add = generic_fallback_add_vertical,
    sub = generic_fallback_sub_vertical,
    mul = generic_fallback_mul_vertical,
    div = generic_fallback_div_vertical,
    Fallback,
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_arithmetic_impls!(
    add = generic_avx2_add_vertical,
    sub = generic_avx2_sub_vertical,
    mul = generic_avx2_mul_vertical,
    div = generic_avx2_div_vertical,
    Avx2,
    target_features = "avx2"
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_arithmetic_impls!(
    add = generic_avx512_add_vertical,
    sub = generic_avx512_sub_vertical,
    mul = generic_avx512_mul_vertical,
    div = generic_avx512_div_vertical,
    Avx512,
    target_features = "avx512f",
    "avx512bw"
);
#[cfg(target_arch = "aarch64")]
define_arithmetic_impls!(
    add = generic_neon_add_vertical,
    sub = generic_neon_sub_vertical,
    mul = generic_neon_mul_vertical,
    div = generic_neon_div_vertical,
    Neon,
    target_features = "neon"
);

#[cfg(test)]
mod tests {
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
                    assert_eq!(
                        result,
                        expected,
                        "Routine result does not match expected",
                    );
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
                    assert_eq!(
                        result,
                        expected,
                        "Routine result does not match expected",
                    );
                }
            }
        };
    }

    macro_rules! define_arithmetic_test {
        ($variant:ident, types = $($t:ident $(,)?)+) => {
            $(
                define_inner_test!($variant, op = add, ty = $t);
                define_inner_test!($variant, op = sub, ty = $t);
                define_inner_test!($variant, op = mul, ty = $t);
                define_inner_test!($variant, op = div, ty = $t);
            )*
        };
    }

    define_arithmetic_test!(
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
    define_arithmetic_test!(
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
    define_arithmetic_test!(
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
    define_arithmetic_test!(
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
