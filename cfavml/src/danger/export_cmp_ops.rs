//! Comparison related operations
//!
//! Although some of these operations i.e. (max_horizontal, min_horizontal) are technically aggregate
//! routines, they are grouped with the rest of their cmp operations for simplicity.

use crate::buffer::WriteOnlyBuffer;
use crate::danger::{
    generic_max_horizontal,
    generic_max_value,
    generic_max_vector,
    generic_min_horizontal,
    generic_min_value,
    generic_min_vector,
    SimdRegister,
};
use crate::math::{AutoMath, Math};

macro_rules! define_max_impls {
    (
        vertical = $max_vector_name:ident,
        value = $max_value_name:ident,
        horizontal = $max_horizontal_name:ident,
        $imp:ident $(,)?
        $(target_features = $($feat:expr $(,)?)+)?
    ) => {
        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = r#"
            Takes the element wise max of two vectors of size `dims` and stores the result
            in `result` of size `dims`.

            ### Pseudocode

            ```ignore
            result = [0; dims]

            for i in range(dims):
                result[i] = max(a[i], b[i])

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
            - The sizes of `a`, `b` and `result` must also be equal to size `dims` otherwise out of
              bounds access can occur.
        "#]
        pub unsafe fn $max_vector_name<T, B>(
            dims: usize,
            a: &[T],
            b: &[T],
            result: &mut [B],
        )
        where
            T: Copy,
            crate::danger::$imp: SimdRegister<T>,
            AutoMath: Math<T>,
            for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
        {
            generic_max_vector::<T, crate::danger::$imp, AutoMath, B>(
                dims,
                a,
                b,
                result,
            )
        }

        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = r#"
            Takes the element wise max of the provided broadcast value and a vector of size `dims`
            and stores the result in `result` of size `dims`.

            ### Pseudocode

            ```ignore
            result = [0; dims]

            for i in range(dims):
                result[i] = max(value, a[i])

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
            - The sizes of `a` and `result` must also be equal to size `dims` otherwise out of
              bounds access can occur.
        "#]
        pub unsafe fn $max_value_name<T, B>(
            dims: usize,
            value: T,
            a: &[T],
            result: &mut [B],
        )
        where
            T: Copy,
            crate::danger::$imp: SimdRegister<T>,
            AutoMath: Math<T>,
            for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
        {
            generic_max_value::<T, crate::danger::$imp, AutoMath, B>(
                dims,
                value,
                a,
                result,
            )
        }


        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = r#"
            Performs a horizontal max of all elements in the provided vector `a` of size `dims`
            returning the max value.

            ### Pseudocode

            ```ignore
            result = -inf

            for i in range(dims):
                result = max(result, a[i])

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
        pub unsafe fn $max_horizontal_name<T>(
            dims: usize,
            a: &[T],
        ) -> T
        where
            T: Copy,
            crate::danger::$imp: SimdRegister<T>,
            AutoMath: Math<T>,
        {
            generic_max_horizontal::<T, crate::danger::$imp, AutoMath>(
                dims,
                a,
            )
        }
    };
}

define_max_impls!(
    vertical = generic_fallback_max_vector,
    value = generic_fallback_max_value,
    horizontal = generic_fallback_max_horizontal,
    Fallback,
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_max_impls!(
    vertical = generic_avx2_max_vector,
    value = generic_avx2_max_value,
    horizontal = generic_avx2_max_horizontal,
    Avx2,
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_max_impls!(
    vertical = generic_avx512_max_vector,
    value = generic_avx512_max_value,
    horizontal = generic_avx512_max_horizontal,
    Avx512,
);
#[cfg(target_arch = "aarch64")]
define_max_impls!(
    vertical = generic_neon_nofma_max_vector,
    value = generic_neon_nofma_max_value,
    horizontal = generic_neon_nofma_max_horizontal,
    Neon,
);

macro_rules! define_min_impls {
    (
        vertical = $min_vector_name:ident,
        value = $min_value_name:ident,
        horizontal = $min_horizontal_name:ident,
        $imp:ident $(,)?
        $(target_features = $($feat:expr $(,)?)+)?
    ) => {
        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = r#"
            Takes the element wise min of two vectors of size `dims` and stores the result
            in `result` of size `dims`.

            ### Pseudocode

            ```ignore
            result = [0; dims]

            for i in range(dims):
                result[i] = min(a[i], b[i])

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
            - The sizes of `a`, `b` and `result` must also be equal to size `dims` otherwise out of
              bounds access can occur.
        "#]
        pub unsafe fn $min_vector_name<T, B>(
            dims: usize,
            a: &[T],
            b: &[T],
            result: &mut [B],
        )
        where
            T: Copy,
            crate::danger::$imp: SimdRegister<T>,
            AutoMath: Math<T>,
            for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
        {
            generic_min_vector::<T, crate::danger::$imp, AutoMath, B>(
                dims,
                a,
                b,
                result,
            )
        }

        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = r#"
            Takes the element wise min of the provided broadcast value and a vector of size `dims`
            and stores the result in `result` of size `dims`.

            ### Pseudocode

            ```ignore
            result = [0; dims]

            for i in range(dims):
                result[i] = min(value, a[i])

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
            - The sizes of `a` and `result` must also be equal to size `dims` otherwise out of
              bounds access can occur.
        "#]
        pub unsafe fn $min_value_name<T, B>(
            dims: usize,
            value: T,
            a: &[T],
            result: &mut [B],
        )
        where
            T: Copy,
            crate::danger::$imp: SimdRegister<T>,
            AutoMath: Math<T>,
            for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
        {
            generic_min_value::<T, crate::danger::$imp, AutoMath, B>(
                dims,
                value,
                a,
                result,
            )
        }


        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = r#"
            Performs a horizontal min of all elements in the provided vector `a` of size `dims`
            returning the min value.

            ### Pseudocode

            ```ignore
            result = -inf

            for i in range(dims):
                result = min(result, a[i])

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
        pub unsafe fn $min_horizontal_name<T>(
            dims: usize,
            a: &[T],
        ) -> T
        where
            T: Copy,
            crate::danger::$imp: SimdRegister<T>,
            AutoMath: Math<T>,
        {
            generic_min_horizontal::<T, crate::danger::$imp, AutoMath>(
                dims,
                a,
            )
        }
    };
}

define_min_impls!(
    vertical = generic_fallback_min_vector,
    value = generic_fallback_min_value,
    horizontal = generic_fallback_min_horizontal,
    Fallback,
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_min_impls!(
    vertical = generic_avx2_min_vector,
    value = generic_avx2_min_value,
    horizontal = generic_avx2_min_horizontal,
    Avx2,
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_min_impls!(
    vertical = generic_avx512_min_vector,
    value = generic_avx512_min_value,
    horizontal = generic_avx512_min_horizontal,
    Avx512,
);
#[cfg(target_arch = "aarch64")]
define_min_impls!(
    vertical = generic_neon_nofma_min_vector,
    value = generic_neon_nofma_min_value,
    horizontal = generic_neon_nofma_min_horizontal,
    Neon,
);

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! define_inner_test {
        ($variant:ident, op = $op:ident, ty = $t:ident, fold_on = $fold_cb:ident) => {
            paste::paste! {
                #[test]
                fn [< $variant _ $op _horizontal_ $t >]() {
                    let (l1, _) = crate::test_utils::get_sample_vectors::<$t>(533);

                    let result = unsafe { [< $variant _ $op _horizontal >](l1.len(), &l1) };

                    let expected = l1.iter()
                        .copied()
                        .fold(AutoMath::$fold_cb(), |a, b| AutoMath::[< cmp_ $op >](a, b));
                    assert_eq!(
                        result,
                        expected,
                        "Routine result does not match expected",
                    );
                }

                #[test]
                fn [< $variant _ $op _value_ $t >]() {
                    let (l1, _) = crate::test_utils::get_sample_vectors::<$t>(533);

                    let mut result = vec![$t::default(); 533];
                    unsafe { [< $variant _ $op _value >](l1.len(), 2 as $t, &l1, &mut result) };

                    let expected = l1.iter()
                        .copied()
                        .map(|v| AutoMath::[< cmp_ $op >](v, 2 as $t))
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
                    unsafe { [< $variant _ $op _vector >](l1.len(), &l1, &l2, &mut result) };

                    let expected = l1.iter()
                        .copied()
                        .zip(l2.iter().copied())
                        .map(|(a, b)| AutoMath::[< cmp_ $op >](a, b))
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

    macro_rules! define_cmp_test {
        ($variant:ident, types = $($t:ident $(,)?)+) => {
            $(
                define_inner_test!($variant, op = min, ty = $t, fold_on = max);
                define_inner_test!($variant, op = max, ty = $t, fold_on = min);
            )*
        };
    }

    define_cmp_test!(
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
    define_cmp_test!(
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
    define_cmp_test!(
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
    define_cmp_test!(
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
