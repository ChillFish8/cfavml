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
