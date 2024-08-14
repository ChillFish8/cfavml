//! Common arithmetic operations
//!
//! I.e. Add, Sub, Mul, Div...

use crate::buffer::WriteOnlyBuffer;
use crate::danger::{
    generic_add_value,
    generic_add_vector,
    generic_div_value,
    generic_div_vector,
    generic_mul_value,
    generic_mul_vector,
    generic_sub_value,
    generic_sub_vector,
    SimdRegister,
};
use crate::math::{AutoMath, Math};

macro_rules! define_arithmetic_impls {
    (
        add_value = $add_value_name:ident,
        add_vector = $add_vector_name:ident,
        sub_value = $sub_value_name:ident,
        sub_vector = $sub_vector_name:ident,
        mul_value = $mul_value_name:ident,
        mul_vector = $mul_vector_name:ident,
        div_value = $div_value_name:ident,
        div_vector = $div_vector_name:ident,
        $imp:ident $(,)?
        $(target_features = $($feat:expr $(,)?)+)?
    ) => {
        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = r#"
            Adds a single value to each element in vector `a` of size `dims`.

            ### Pseudocode

            ```ignore
            result = [0; dims]

            for i in range(dims):
                result[i] = a[i] + value

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
        pub unsafe fn $add_value_name<T, B>(
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
            generic_add_value::<T, crate::danger::$imp, AutoMath, B>(
                dims,
                value,
                a,
                result,
            )
        }

        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = r#"
            Performs an element wise add of `a` and `b` of size `dims` and store the result
            in `result` vector of size `dims`.

            ### Pseudocode

            ```ignore
            result = [0; dims]

            for i in range(dims):
                result[i] = a[i] + b[i]

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
        pub unsafe fn $add_vector_name<T, B>(
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
            generic_add_vector::<T, crate::danger::$imp, AutoMath, B>(
                dims,
                a,
                b,
                result,
            )
        }

        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = r#"
            Subtracts a single value from each element in vector `a` of size `dims`.

            ### Pseudocode

            ```ignore
            result = [0; dims]

            for i in range(dims):
                result[i] = a[i] - value

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
        pub unsafe fn $sub_value_name<T, B>(
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
            generic_sub_value::<T, crate::danger::$imp, AutoMath, B>(
                dims,
                value,
                a,
                result,
            )
        }

        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = r#"
            Performs an element wise subtraction of `a` and `b` of size `dims` and store the result
            in `result` vector of size `dims`.

            ### Pseudocode

            ```ignore
            result = [0; dims]

            for i in range(dims):
                result[i] = a[i] - b[i]

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
        pub unsafe fn $sub_vector_name<T, B>(
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
            generic_sub_vector::<T, crate::danger::$imp, AutoMath, B>(
                dims,
                a,
                b,
                result,
            )
        }

        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = r#"
            Multiplies each element in vector `a` of size `dims` by `value`.

            ### Pseudocode

            ```ignore
            result = [0; dims]

            for i in range(dims):
                result[i] = a[i] * value

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
        pub unsafe fn $mul_value_name<T, B>(
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
            generic_mul_value::<T, crate::danger::$imp, AutoMath, B>(
                dims,
                value,
                a,
                result,
            )
        }

        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = r#"
            Performs an element wise multiplication of `a` and `b` of size `dims` and store the result
            in `result` vector of size `dims`.

            ### Pseudocode

            ```ignore
            result = [0; dims]

            for i in range(dims):
                result[i] = a[i] * b[i]

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
        pub unsafe fn $mul_vector_name<T, B>(
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
            generic_mul_vector::<T, crate::danger::$imp, AutoMath, B>(
                dims,
                a,
                b,
                result,
            )
        }

        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = r#"
            Divides each element in vector `a` of size `dims` by `value`.

            ### Pseudocode

            ```ignore
            result = [0; dims]

            for i in range(dims):
                result[i] = a[i] / value

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
        pub unsafe fn $div_value_name<T, B>(
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
            generic_div_value::<T, crate::danger::$imp, AutoMath, B>(
                dims,
                value,
                a,
                result,
            )
        }

        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = r#"
            Performs an element wise division of `a` and `b` of size `dims` and store the result
            in `result` vector of size `dims`.

            ### Pseudocode

            ```ignore
            result = [0; dims]

            for i in range(dims):
                result[i] = a[i] / b[i]

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
        pub unsafe fn $div_vector_name<T, B>(
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
            generic_div_vector::<T, crate::danger::$imp, AutoMath, B>(
                dims,
                a,
                b,
                result,
            )
        }
    };
}

define_arithmetic_impls!(
    add_value = generic_fallback_add_value,
    add_vector = generic_fallback_add_vector,
    sub_value = generic_fallback_sub_value,
    sub_vector = generic_fallback_sub_vector,
    mul_value = generic_fallback_mul_value,
    mul_vector = generic_fallback_mul_vector,
    div_value = generic_fallback_div_value,
    div_vector = generic_fallback_div_vector,
    Fallback,
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_arithmetic_impls!(
    add_value = generic_avx2_add_value,
    add_vector = generic_avx2_add_vector,
    sub_value = generic_avx2_sub_value,
    sub_vector = generic_avx2_sub_vector,
    mul_value = generic_avx2_mul_value,
    mul_vector = generic_avx2_mul_vector,
    div_value = generic_avx2_div_value,
    div_vector = generic_avx2_div_vector,
    Avx2,
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_arithmetic_impls!(
    add_value = generic_avx512_add_value,
    add_vector = generic_avx512_add_vector,
    sub_value = generic_avx512_sub_value,
    sub_vector = generic_avx512_sub_vector,
    mul_value = generic_avx512_mul_value,
    mul_vector = generic_avx512_mul_vector,
    div_value = generic_avx512_div_value,
    div_vector = generic_avx512_div_vector,
    Avx512,
);
#[cfg(target_arch = "aarch64")]
define_arithmetic_impls!(
    add_value = generic_neon_nofma_add_value,
    add_vector = generic_neon_nofma_add_vector,
    sub_value = generic_neon_nofma_sub_value,
    sub_vector = generic_neon_nofma_sub_vector,
    mul_value = generic_neon_nofma_mul_value,
    mul_vector = generic_neon_nofma_mul_vector,
    div_value = generic_neon_nofma_div_value,
    div_vector = generic_neon_nofma_div_vector,
    Neon,
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
                    unsafe { [< $variant _ $op _value >](l1.len(), 2 as $t, &l1, &mut result) };

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
                    unsafe { [< $variant _ $op _vector >](l1.len(), &l1, &l2, &mut result) };

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
