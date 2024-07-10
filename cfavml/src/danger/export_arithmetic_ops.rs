//! Generated non-generic methods for SIMD based arithmetic operations.
//!
//! Although the generic methods can provide greater ergonomics within your code
//! they do not apply the necessary feature assumptions or type ergonomics, that
//! is the purpose of these exported methods which ultimately end up calling the
//! generic variant.
//!
//! Method names are exported in the following format:
//! ```no_test
//! <type>_x<const/any>_<arch>_<fma/nofma>_<op>
//! ```
//!
//! ### `xconst` vs `xany`
//!
//! We export const generic versions of the impls which allow the compiler to perform further
//! optimizations on the routines like unrolling and tail loop elimination, if you are using
//! these routines where you know the vector dimensions at compile time, it is probably worth
//! using the `xconst` variants, otherwise use the `xany` methods.
//!

use super::*;
use crate::math::AutoMath;

macro_rules! export_vector_x_value_op {
    (
        description = $desc:expr,
        ty = $t:ident,
        register = $im:ident,
        op = $op:ident,
        xconst = $xconst_name:ident,
        xany = $xany_name:ident,
    ) => {
        #[doc = concat!("`", stringify!($t), "` ", $desc)]
        ///
        /// # Safety
        ///
        /// Vectors **`a`** and **`result`** must be equal in length to that specified in **`DIMS`**,
        /// otherwise out of bound access can occur.
        pub unsafe fn $xconst_name<const DIMS: usize>(
            value: $t,
            a: &[$t],
            result: &mut [$t],
        ) {
            $op::<_, $im, AutoMath>(DIMS, value, a, result)
        }

        #[doc = concat!("`", stringify!($t), "` ", $desc)]
        ///
        /// # Safety
        ///
        /// Vectors **`a`** and **`result`** must be equal in length otherwise out of bound
        /// access can occur.
        pub unsafe fn $xany_name(
            value: $t,
            a: &[$t],
            result: &mut [$t],
        ) {
            $op::<_, $im, AutoMath>(a.len(), value, a, result)
        }
    };
    (
        description = $desc:expr,
        ty = $t:ident,
        register = $im:ident,
        op = $op:ident,
        xconst = $xconst_name:ident,
        xany = $xany_name:ident,
        features = $($feat:expr $(,)?)*
    ) => {
        #[target_feature($(enable = $feat , )*)]
        #[doc = concat!("`", stringify!($t), "` ", $desc)]
        #[doc = "\n\n# Safety\n\n"]
        #[doc = concat!("The following CPU flags must be available: ", $("**", $feat, "**", ", ",)*)]
        ///
        /// Vectors **`a`** and **`result`** must be equal in length to that specified in **`DIMS`**,
        /// otherwise out of bound access can occur.
        pub unsafe fn $xconst_name<const DIMS: usize>(
            value: $t,
            a: &[$t],
            result: &mut [$t],
        ) {
            $op::<_, $im, AutoMath>(DIMS, value, a, result)
        }

        #[target_feature($(enable = $feat , )*)]
        #[doc = concat!("`", stringify!($t), "` ", $desc)]
        #[doc = "\n\n# Safety\n\n"]
        #[doc = concat!("The following CPU flags must be available: ", $("**", $feat, "**", ", ",)*)]
        ///
        /// Vectors **`a`** and **`result`** must be equal in length otherwise out of bound
        /// access can occur.
        pub unsafe fn $xany_name(
            value: $t,
            a: &[$t],
            result: &mut [$t],
        ) {
            $op::<_, $im, AutoMath>(a.len(), value, a, result)
        }
    };
}

macro_rules! export_vector_x_vector_op {
    (
        description = $desc:expr,
        ty = $t:ident,
        register = $im:ident,
        op = $op:ident,
        xconst = $xconst_name:ident,
        xany = $xany_name:ident,
    ) => {
        #[doc = concat!("`", stringify!($t), "` ", $desc)]
        ///
        /// # Safety
        ///
        /// Vectors **`a`**, **`b`** and **`result`** must be equal in length to that specified in **`DIMS`**,
        /// otherwise out of bound access can occur.
        pub unsafe fn $xconst_name<const DIMS: usize>(
            a: &[$t],
            b: &[$t],
            result: &mut [$t],
        ) {
            $op::<_, $im, AutoMath>(DIMS, a, b, result)
        }

        #[doc = concat!("`", stringify!($t), "` ", $desc)]
        ///
        /// # Safety
        ///
        /// Vectors **`a`**, **`b`** and **`result`** must be equal in length otherwise out of bound
        /// access can occur.
        pub unsafe fn $xany_name(
            a: &[$t],
            b: &[$t],
            result: &mut [$t],
        ) {
            $op::<_, $im, AutoMath>(a.len(), a, b, result)
        }
    };
    (
        description = $desc:expr,
        ty = $t:ident,
        register = $im:ident,
        op = $op:ident,
        xconst = $xconst_name:ident,
        xany = $xany_name:ident,
        features = $($feat:expr $(,)?)*
    ) => {
        #[target_feature($(enable = $feat , )*)]
        #[doc = concat!("`", stringify!($t), "` ", $desc)]
        #[doc = "\n\n# Safety\n\n"]
        #[doc = concat!("The following CPU flags must be available: ", $("**", $feat, "**", ", ",)*)]
        ///
        /// Vectors **`a`**, **`b`** and **`result`** must be equal in length to that specified in **`DIMS`**,
        /// otherwise out of bound access can occur.
        pub unsafe fn $xconst_name<const DIMS: usize>(
            a: &[$t],
            b: &[$t],
            result: &mut [$t],
        ) {
            $op::<_, $im, AutoMath>(DIMS, a, b, result)
        }

        #[target_feature($(enable = $feat , )*)]
        #[doc = concat!("`", stringify!($t), "` ", $desc)]
        #[doc = "\n\n# Safety\n\n"]
        #[doc = concat!("The following CPU flags must be available: ", $("**", $feat, "**", ", ",)*)]
        ///
        /// Vectors **`a`**, **`b`** and **`result`** must be equal in length otherwise out of bound
        /// access can occur.
        pub unsafe fn $xany_name(
            a: &[$t],
            b: &[$t],
            result: &mut [$t],
        ) {
            $op::<_, $im, AutoMath>(a.len(), a, b, result)
        }
    };
}

/// Vector space distance ops on the fallback implementations.
///
/// These methods do not strictly require any CPU feature but do auto-vectorize
/// well when target features are explicitly provided and can be used to
/// provide accelerated operations on non-explicitly supported SIMD architectures.
pub mod arithmetic_ops_fallback {
    use super::*;

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = f32,
        register = Fallback,
        op = generic_add_value,
        xconst = f32_xconst_fallback_nofma_add_value,
        xany = f32_xany_fallback_nofma_add_value,
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = f32,
        register = Fallback,
        op = generic_sub_value,
        xconst = f32_xconst_fallback_nofma_sub_value,
        xany = f32_xany_fallback_nofma_sub_value,
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = f32,
        register = Fallback,
        op = generic_mul_value,
        xconst = f32_xconst_fallback_nofma_mul_value,
        xany = f32_xany_fallback_nofma_mul_value,
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = f32,
        register = Fallback,
        op = generic_div_value,
        xconst = f32_xconst_fallback_nofma_div_value,
        xany = f32_xany_fallback_nofma_div_value,
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = f32,
        register = Fallback,
        op = generic_add_vector,
        xconst = f32_xconst_fallback_nofma_add_vector,
        xany = f32_xany_fallback_nofma_add_vector,
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = f32,
        register = Fallback,
        op = generic_sub_vector,
        xconst = f32_xconst_fallback_nofma_sub_vector,
        xany = f32_xany_fallback_nofma_sub_vector,
    );
    export_vector_x_vector_op!(
        description = "Vector multiplication, writing to a result vector",
        ty = f32,
        register = Fallback,
        op = generic_mul_vector,
        xconst = f32_xconst_fallback_nofma_mul_vector,
        xany = f32_xany_fallback_nofma_mul_vector,
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = f32,
        register = Fallback,
        op = generic_div_vector,
        xconst = f32_xconst_fallback_nofma_div_vector,
        xany = f32_xany_fallback_nofma_div_vector,
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = f64,
        register = Fallback,
        op = generic_add_value,
        xconst = f64_xconst_fallback_nofma_add_value,
        xany = f64_xany_fallback_nofma_add_value,
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = f64,
        register = Fallback,
        op = generic_sub_value,
        xconst = f64_xconst_fallback_nofma_sub_value,
        xany = f64_xany_fallback_nofma_sub_value,
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = f64,
        register = Fallback,
        op = generic_mul_value,
        xconst = f64_xconst_fallback_nofma_mul_value,
        xany = f64_xany_fallback_nofma_mul_value,
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = f64,
        register = Fallback,
        op = generic_div_value,
        xconst = f64_xconst_fallback_nofma_div_value,
        xany = f64_xany_fallback_nofma_div_value,
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = f64,
        register = Fallback,
        op = generic_add_vector,
        xconst = f64_xconst_fallback_nofma_add_vector,
        xany = f64_xany_fallback_nofma_add_vector,
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = f64,
        register = Fallback,
        op = generic_sub_vector,
        xconst = f64_xconst_fallback_nofma_sub_vector,
        xany = f64_xany_fallback_nofma_sub_vector,
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = f64,
        register = Fallback,
        op = generic_mul_vector,
        xconst = f64_xconst_fallback_nofma_mul_vector,
        xany = f64_xany_fallback_nofma_mul_vector,
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = f64,
        register = Fallback,
        op = generic_div_vector,
        xconst = f64_xconst_fallback_nofma_div_vector,
        xany = f64_xany_fallback_nofma_div_vector,
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = i8,
        register = Fallback,
        op = generic_add_value,
        xconst = i8_xconst_fallback_nofma_add_value,
        xany = i8_xany_fallback_nofma_add_value,
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = i8,
        register = Fallback,
        op = generic_sub_value,
        xconst = i8_xconst_fallback_nofma_sub_value,
        xany = i8_xany_fallback_nofma_sub_value,
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = i8,
        register = Fallback,
        op = generic_mul_value,
        xconst = i8_xconst_fallback_nofma_mul_value,
        xany = i8_xany_fallback_nofma_mul_value,
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = i8,
        register = Fallback,
        op = generic_div_value,
        xconst = i8_xconst_fallback_nofma_div_value,
        xany = i8_xany_fallback_nofma_div_value,
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = i8,
        register = Fallback,
        op = generic_add_vector,
        xconst = i8_xconst_fallback_nofma_add_vector,
        xany = i8_xany_fallback_nofma_add_vector,
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = i8,
        register = Fallback,
        op = generic_sub_vector,
        xconst = i8_xconst_fallback_nofma_sub_vector,
        xany = i8_xany_fallback_nofma_sub_vector,
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = i8,
        register = Fallback,
        op = generic_mul_vector,
        xconst = i8_xconst_fallback_nofma_mul_vector,
        xany = i8_xany_fallback_nofma_mul_vector,
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = i8,
        register = Fallback,
        op = generic_div_vector,
        xconst = i8_xconst_fallback_nofma_div_vector,
        xany = i8_xany_fallback_nofma_div_vector,
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = i16,
        register = Fallback,
        op = generic_add_value,
        xconst = i16_xconst_fallback_nofma_add_value,
        xany = i16_xany_fallback_nofma_add_value,
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = i16,
        register = Fallback,
        op = generic_sub_value,
        xconst = i16_xconst_fallback_nofma_sub_value,
        xany = i16_xany_fallback_nofma_sub_value,
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = i16,
        register = Fallback,
        op = generic_mul_value,
        xconst = i16_xconst_fallback_nofma_mul_value,
        xany = i16_xany_fallback_nofma_mul_value,
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = i16,
        register = Fallback,
        op = generic_div_value,
        xconst = i16_xconst_fallback_nofma_div_value,
        xany = i16_xany_fallback_nofma_div_value,
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = i16,
        register = Fallback,
        op = generic_add_vector,
        xconst = i16_xconst_fallback_nofma_add_vector,
        xany = i16_xany_fallback_nofma_add_vector,
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = i16,
        register = Fallback,
        op = generic_sub_vector,
        xconst = i16_xconst_fallback_nofma_sub_vector,
        xany = i16_xany_fallback_nofma_sub_vector,
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = i16,
        register = Fallback,
        op = generic_mul_vector,
        xconst = i16_xconst_fallback_nofma_mul_vector,
        xany = i16_xany_fallback_nofma_mul_vector,
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = i16,
        register = Fallback,
        op = generic_div_vector,
        xconst = i16_xconst_fallback_nofma_div_vector,
        xany = i16_xany_fallback_nofma_div_vector,
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = i32,
        register = Fallback,
        op = generic_add_value,
        xconst = i32_xconst_fallback_nofma_add_value,
        xany = i32_xany_fallback_nofma_add_value,
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = i32,
        register = Fallback,
        op = generic_sub_value,
        xconst = i32_xconst_fallback_nofma_sub_value,
        xany = i32_xany_fallback_nofma_sub_value,
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = i32,
        register = Fallback,
        op = generic_mul_value,
        xconst = i32_xconst_fallback_nofma_mul_value,
        xany = i32_xany_fallback_nofma_mul_value,
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = i32,
        register = Fallback,
        op = generic_div_value,
        xconst = i32_xconst_fallback_nofma_div_value,
        xany = i32_xany_fallback_nofma_div_value,
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = i32,
        register = Fallback,
        op = generic_add_vector,
        xconst = i32_xconst_fallback_nofma_add_vector,
        xany = i32_xany_fallback_nofma_add_vector,
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = i32,
        register = Fallback,
        op = generic_sub_vector,
        xconst = i32_xconst_fallback_nofma_sub_vector,
        xany = i32_xany_fallback_nofma_sub_vector,
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = i32,
        register = Fallback,
        op = generic_mul_vector,
        xconst = i32_xconst_fallback_nofma_mul_vector,
        xany = i32_xany_fallback_nofma_mul_vector,
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = i32,
        register = Fallback,
        op = generic_div_vector,
        xconst = i32_xconst_fallback_nofma_div_vector,
        xany = i32_xany_fallback_nofma_div_vector,
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = i64,
        register = Fallback,
        op = generic_add_value,
        xconst = i64_xconst_fallback_nofma_add_value,
        xany = i64_xany_fallback_nofma_add_value,
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = i64,
        register = Fallback,
        op = generic_sub_value,
        xconst = i64_xconst_fallback_nofma_sub_value,
        xany = i64_xany_fallback_nofma_sub_value,
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = i64,
        register = Fallback,
        op = generic_mul_value,
        xconst = i64_xconst_fallback_nofma_mul_value,
        xany = i64_xany_fallback_nofma_mul_value,
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = i64,
        register = Fallback,
        op = generic_div_value,
        xconst = i64_xconst_fallback_nofma_div_value,
        xany = i64_xany_fallback_nofma_div_value,
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = i64,
        register = Fallback,
        op = generic_add_vector,
        xconst = i64_xconst_fallback_nofma_add_vector,
        xany = i64_xany_fallback_nofma_add_vector,
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = i64,
        register = Fallback,
        op = generic_sub_vector,
        xconst = i64_xconst_fallback_nofma_sub_vector,
        xany = i64_xany_fallback_nofma_sub_vector,
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = i64,
        register = Fallback,
        op = generic_mul_vector,
        xconst = i64_xconst_fallback_nofma_mul_vector,
        xany = i64_xany_fallback_nofma_mul_vector,
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = i64,
        register = Fallback,
        op = generic_div_vector,
        xconst = i64_xconst_fallback_nofma_div_vector,
        xany = i64_xany_fallback_nofma_div_vector,
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = u8,
        register = Fallback,
        op = generic_add_value,
        xconst = u8_xconst_fallback_nofma_add_value,
        xany = u8_xany_fallback_nofma_add_value,
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = u8,
        register = Fallback,
        op = generic_sub_value,
        xconst = u8_xconst_fallback_nofma_sub_value,
        xany = u8_xany_fallback_nofma_sub_value,
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = u8,
        register = Fallback,
        op = generic_mul_value,
        xconst = u8_xconst_fallback_nofma_mul_value,
        xany = u8_xany_fallback_nofma_mul_value,
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = u8,
        register = Fallback,
        op = generic_div_value,
        xconst = u8_xconst_fallback_nofma_div_value,
        xany = u8_xany_fallback_nofma_div_value,
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = u8,
        register = Fallback,
        op = generic_add_vector,
        xconst = u8_xconst_fallback_nofma_add_vector,
        xany = u8_xany_fallback_nofma_add_vector,
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = u8,
        register = Fallback,
        op = generic_sub_vector,
        xconst = u8_xconst_fallback_nofma_sub_vector,
        xany = u8_xany_fallback_nofma_sub_vector,
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = u8,
        register = Fallback,
        op = generic_mul_vector,
        xconst = u8_xconst_fallback_nofma_mul_vector,
        xany = u8_xany_fallback_nofma_mul_vector,
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = u8,
        register = Fallback,
        op = generic_div_vector,
        xconst = u8_xconst_fallback_nofma_div_vector,
        xany = u8_xany_fallback_nofma_div_vector,
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = u16,
        register = Fallback,
        op = generic_add_value,
        xconst = u16_xconst_fallback_nofma_add_value,
        xany = u16_xany_fallback_nofma_add_value,
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = u16,
        register = Fallback,
        op = generic_sub_value,
        xconst = u16_xconst_fallback_nofma_sub_value,
        xany = u16_xany_fallback_nofma_sub_value,
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = u16,
        register = Fallback,
        op = generic_mul_value,
        xconst = u16_xconst_fallback_nofma_mul_value,
        xany = u16_xany_fallback_nofma_mul_value,
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = u16,
        register = Fallback,
        op = generic_div_value,
        xconst = u16_xconst_fallback_nofma_div_value,
        xany = u16_xany_fallback_nofma_div_value,
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = u16,
        register = Fallback,
        op = generic_add_vector,
        xconst = u16_xconst_fallback_nofma_add_vector,
        xany = u16_xany_fallback_nofma_add_vector,
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = u16,
        register = Fallback,
        op = generic_sub_vector,
        xconst = u16_xconst_fallback_nofma_sub_vector,
        xany = u16_xany_fallback_nofma_sub_vector,
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = u16,
        register = Fallback,
        op = generic_mul_vector,
        xconst = u16_xconst_fallback_nofma_mul_vector,
        xany = u16_xany_fallback_nofma_mul_vector,
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = u16,
        register = Fallback,
        op = generic_div_vector,
        xconst = u16_xconst_fallback_nofma_div_vector,
        xany = u16_xany_fallback_nofma_div_vector,
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = u32,
        register = Fallback,
        op = generic_add_value,
        xconst = u32_xconst_fallback_nofma_add_value,
        xany = u32_xany_fallback_nofma_add_value,
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = u32,
        register = Fallback,
        op = generic_sub_value,
        xconst = u32_xconst_fallback_nofma_sub_value,
        xany = u32_xany_fallback_nofma_sub_value,
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = u32,
        register = Fallback,
        op = generic_mul_value,
        xconst = u32_xconst_fallback_nofma_mul_value,
        xany = u32_xany_fallback_nofma_mul_value,
    );
    export_vector_x_value_op!(
        description = "Value division, writing to a result vector",
        ty = u32,
        register = Fallback,
        op = generic_div_value,
        xconst = u32_xconst_fallback_nofma_div_value,
        xany = u32_xany_fallback_nofma_div_value,
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = u32,
        register = Fallback,
        op = generic_add_vector,
        xconst = u32_xconst_fallback_nofma_add_vector,
        xany = u32_xany_fallback_nofma_add_vector,
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = u32,
        register = Fallback,
        op = generic_sub_vector,
        xconst = u32_xconst_fallback_nofma_sub_vector,
        xany = u32_xany_fallback_nofma_sub_vector,
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = u32,
        register = Fallback,
        op = generic_mul_vector,
        xconst = u32_xconst_fallback_nofma_mul_vector,
        xany = u32_xany_fallback_nofma_mul_vector,
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = u32,
        register = Fallback,
        op = generic_div_vector,
        xconst = u32_xconst_fallback_nofma_div_vector,
        xany = u32_xany_fallback_nofma_div_vector,
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = u64,
        register = Fallback,
        op = generic_add_value,
        xconst = u64_xconst_fallback_nofma_add_value,
        xany = u64_xany_fallback_nofma_add_value,
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = u64,
        register = Fallback,
        op = generic_sub_value,
        xconst = u64_xconst_fallback_nofma_sub_value,
        xany = u64_xany_fallback_nofma_sub_value,
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = u64,
        register = Fallback,
        op = generic_mul_value,
        xconst = u64_xconst_fallback_nofma_mul_value,
        xany = u64_xany_fallback_nofma_mul_value,
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = u64,
        register = Fallback,
        op = generic_div_value,
        xconst = u64_xconst_fallback_nofma_div_value,
        xany = u64_xany_fallback_nofma_div_value,
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = u64,
        register = Fallback,
        op = generic_add_vector,
        xconst = u64_xconst_fallback_nofma_add_vector,
        xany = u64_xany_fallback_nofma_add_vector,
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = u64,
        register = Fallback,
        op = generic_sub_vector,
        xconst = u64_xconst_fallback_nofma_sub_vector,
        xany = u64_xany_fallback_nofma_sub_vector,
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = u64,
        register = Fallback,
        op = generic_mul_vector,
        xconst = u64_xconst_fallback_nofma_mul_vector,
        xany = u64_xany_fallback_nofma_mul_vector,
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = u64,
        register = Fallback,
        op = generic_div_vector,
        xconst = u64_xconst_fallback_nofma_div_vector,
        xany = u64_xany_fallback_nofma_div_vector,
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
/// Vector space distance ops on the AVX2 supported architectures.
///
/// These methods **must** have AVX2 available to them at the time of calling
/// otherwise all of these methods become UB even if the input data is OK.
pub mod arithmetic_ops_avx2 {
    use super::*;

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = f32,
        register = Avx2,
        op = generic_add_value,
        xconst = f32_xconst_avx2_nofma_add_value,
        xany = f32_xany_avx2_nofma_add_value,
        features = "avx2"
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = f32,
        register = Avx2,
        op = generic_sub_value,
        xconst = f32_xconst_avx2_nofma_sub_value,
        xany = f32_xany_avx2_nofma_sub_value,
        features = "avx2"
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = f32,
        register = Avx2,
        op = generic_mul_value,
        xconst = f32_xconst_avx2_nofma_mul_value,
        xany = f32_xany_avx2_nofma_mul_value,
        features = "avx2"
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = f32,
        register = Avx2,
        op = generic_div_value,
        xconst = f32_xconst_avx2_nofma_div_value,
        xany = f32_xany_avx2_nofma_div_value,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = f32,
        register = Avx2,
        op = generic_add_vector,
        xconst = f32_xconst_avx2_nofma_add_vector,
        xany = f32_xany_avx2_nofma_add_vector,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = f32,
        register = Avx2,
        op = generic_sub_vector,
        xconst = f32_xconst_avx2_nofma_sub_vector,
        xany = f32_xany_avx2_nofma_sub_vector,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector multiplication, writing to a result vector",
        ty = f32,
        register = Avx2,
        op = generic_mul_vector,
        xconst = f32_xconst_avx2_nofma_mul_vector,
        xany = f32_xany_avx2_nofma_mul_vector,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = f32,
        register = Avx2,
        op = generic_div_vector,
        xconst = f32_xconst_avx2_nofma_div_vector,
        xany = f32_xany_avx2_nofma_div_vector,
        features = "avx2"
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = f64,
        register = Avx2,
        op = generic_add_value,
        xconst = f64_xconst_avx2_nofma_add_value,
        xany = f64_xany_avx2_nofma_add_value,
        features = "avx2"
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = f64,
        register = Avx2,
        op = generic_sub_value,
        xconst = f64_xconst_avx2_nofma_sub_value,
        xany = f64_xany_avx2_nofma_sub_value,
        features = "avx2"
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = f64,
        register = Avx2,
        op = generic_mul_value,
        xconst = f64_xconst_avx2_nofma_mul_value,
        xany = f64_xany_avx2_nofma_mul_value,
        features = "avx2"
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = f64,
        register = Avx2,
        op = generic_div_value,
        xconst = f64_xconst_avx2_nofma_div_value,
        xany = f64_xany_avx2_nofma_div_value,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = f64,
        register = Avx2,
        op = generic_add_vector,
        xconst = f64_xconst_avx2_nofma_add_vector,
        xany = f64_xany_avx2_nofma_add_vector,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = f64,
        register = Avx2,
        op = generic_sub_vector,
        xconst = f64_xconst_avx2_nofma_sub_vector,
        xany = f64_xany_avx2_nofma_sub_vector,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = f64,
        register = Avx2,
        op = generic_mul_vector,
        xconst = f64_xconst_avx2_nofma_mul_vector,
        xany = f64_xany_avx2_nofma_mul_vector,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = f64,
        register = Avx2,
        op = generic_div_vector,
        xconst = f64_xconst_avx2_nofma_div_vector,
        xany = f64_xany_avx2_nofma_div_vector,
        features = "avx2"
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = i8,
        register = Avx2,
        op = generic_add_value,
        xconst = i8_xconst_avx2_nofma_add_value,
        xany = i8_xany_avx2_nofma_add_value,
        features = "avx2"
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = i8,
        register = Avx2,
        op = generic_sub_value,
        xconst = i8_xconst_avx2_nofma_sub_value,
        xany = i8_xany_avx2_nofma_sub_value,
        features = "avx2"
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = i8,
        register = Avx2,
        op = generic_mul_value,
        xconst = i8_xconst_avx2_nofma_mul_value,
        xany = i8_xany_avx2_nofma_mul_value,
        features = "avx2"
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = i8,
        register = Avx2,
        op = generic_div_value,
        xconst = i8_xconst_avx2_nofma_div_value,
        xany = i8_xany_avx2_nofma_div_value,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = i8,
        register = Avx2,
        op = generic_add_vector,
        xconst = i8_xconst_avx2_nofma_add_vector,
        xany = i8_xany_avx2_nofma_add_vector,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = i8,
        register = Avx2,
        op = generic_sub_vector,
        xconst = i8_xconst_avx2_nofma_sub_vector,
        xany = i8_xany_avx2_nofma_sub_vector,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = i8,
        register = Avx2,
        op = generic_mul_vector,
        xconst = i8_xconst_avx2_nofma_mul_vector,
        xany = i8_xany_avx2_nofma_mul_vector,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = i8,
        register = Avx2,
        op = generic_div_vector,
        xconst = i8_xconst_avx2_nofma_div_vector,
        xany = i8_xany_avx2_nofma_div_vector,
        features = "avx2"
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = i16,
        register = Avx2,
        op = generic_add_value,
        xconst = i16_xconst_avx2_nofma_add_value,
        xany = i16_xany_avx2_nofma_add_value,
        features = "avx2"
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = i16,
        register = Avx2,
        op = generic_sub_value,
        xconst = i16_xconst_avx2_nofma_sub_value,
        xany = i16_xany_avx2_nofma_sub_value,
        features = "avx2"
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = i16,
        register = Avx2,
        op = generic_mul_value,
        xconst = i16_xconst_avx2_nofma_mul_value,
        xany = i16_xany_avx2_nofma_mul_value,
        features = "avx2"
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = i16,
        register = Avx2,
        op = generic_div_value,
        xconst = i16_xconst_avx2_nofma_div_value,
        xany = i16_xany_avx2_nofma_div_value,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = i16,
        register = Avx2,
        op = generic_add_vector,
        xconst = i16_xconst_avx2_nofma_add_vector,
        xany = i16_xany_avx2_nofma_add_vector,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = i16,
        register = Avx2,
        op = generic_sub_vector,
        xconst = i16_xconst_avx2_nofma_sub_vector,
        xany = i16_xany_avx2_nofma_sub_vector,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = i16,
        register = Avx2,
        op = generic_mul_vector,
        xconst = i16_xconst_avx2_nofma_mul_vector,
        xany = i16_xany_avx2_nofma_mul_vector,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = i16,
        register = Avx2,
        op = generic_div_vector,
        xconst = i16_xconst_avx2_nofma_div_vector,
        xany = i16_xany_avx2_nofma_div_vector,
        features = "avx2"
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = i32,
        register = Avx2,
        op = generic_add_value,
        xconst = i32_xconst_avx2_nofma_add_value,
        xany = i32_xany_avx2_nofma_add_value,
        features = "avx2"
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = i32,
        register = Avx2,
        op = generic_sub_value,
        xconst = i32_xconst_avx2_nofma_sub_value,
        xany = i32_xany_avx2_nofma_sub_value,
        features = "avx2"
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = i32,
        register = Avx2,
        op = generic_mul_value,
        xconst = i32_xconst_avx2_nofma_mul_value,
        xany = i32_xany_avx2_nofma_mul_value,
        features = "avx2"
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = i32,
        register = Avx2,
        op = generic_div_value,
        xconst = i32_xconst_avx2_nofma_div_value,
        xany = i32_xany_avx2_nofma_div_value,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = i32,
        register = Avx2,
        op = generic_add_vector,
        xconst = i32_xconst_avx2_nofma_add_vector,
        xany = i32_xany_avx2_nofma_add_vector,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = i32,
        register = Avx2,
        op = generic_sub_vector,
        xconst = i32_xconst_avx2_nofma_sub_vector,
        xany = i32_xany_avx2_nofma_sub_vector,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = i32,
        register = Avx2,
        op = generic_mul_vector,
        xconst = i32_xconst_avx2_nofma_mul_vector,
        xany = i32_xany_avx2_nofma_mul_vector,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = i32,
        register = Avx2,
        op = generic_div_vector,
        xconst = i32_xconst_avx2_nofma_div_vector,
        xany = i32_xany_avx2_nofma_div_vector,
        features = "avx2"
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = i64,
        register = Avx2,
        op = generic_add_value,
        xconst = i64_xconst_avx2_nofma_add_value,
        xany = i64_xany_avx2_nofma_add_value,
        features = "avx2"
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = i64,
        register = Avx2,
        op = generic_sub_value,
        xconst = i64_xconst_avx2_nofma_sub_value,
        xany = i64_xany_avx2_nofma_sub_value,
        features = "avx2"
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = i64,
        register = Avx2,
        op = generic_mul_value,
        xconst = i64_xconst_avx2_nofma_mul_value,
        xany = i64_xany_avx2_nofma_mul_value,
        features = "avx2"
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = i64,
        register = Avx2,
        op = generic_div_value,
        xconst = i64_xconst_avx2_nofma_div_value,
        xany = i64_xany_avx2_nofma_div_value,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = i64,
        register = Avx2,
        op = generic_add_vector,
        xconst = i64_xconst_avx2_nofma_add_vector,
        xany = i64_xany_avx2_nofma_add_vector,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = i64,
        register = Avx2,
        op = generic_sub_vector,
        xconst = i64_xconst_avx2_nofma_sub_vector,
        xany = i64_xany_avx2_nofma_sub_vector,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = i64,
        register = Avx2,
        op = generic_mul_vector,
        xconst = i64_xconst_avx2_nofma_mul_vector,
        xany = i64_xany_avx2_nofma_mul_vector,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = i64,
        register = Avx2,
        op = generic_div_vector,
        xconst = i64_xconst_avx2_nofma_div_vector,
        xany = i64_xany_avx2_nofma_div_vector,
        features = "avx2"
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = u8,
        register = Avx2,
        op = generic_add_value,
        xconst = u8_xconst_avx2_nofma_add_value,
        xany = u8_xany_avx2_nofma_add_value,
        features = "avx2"
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = u8,
        register = Avx2,
        op = generic_sub_value,
        xconst = u8_xconst_avx2_nofma_sub_value,
        xany = u8_xany_avx2_nofma_sub_value,
        features = "avx2"
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = u8,
        register = Avx2,
        op = generic_mul_value,
        xconst = u8_xconst_avx2_nofma_mul_value,
        xany = u8_xany_avx2_nofma_mul_value,
        features = "avx2"
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = u8,
        register = Avx2,
        op = generic_div_value,
        xconst = u8_xconst_avx2_nofma_div_value,
        xany = u8_xany_avx2_nofma_div_value,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = u8,
        register = Avx2,
        op = generic_add_vector,
        xconst = u8_xconst_avx2_nofma_add_vector,
        xany = u8_xany_avx2_nofma_add_vector,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = u8,
        register = Avx2,
        op = generic_sub_vector,
        xconst = u8_xconst_avx2_nofma_sub_vector,
        xany = u8_xany_avx2_nofma_sub_vector,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = u8,
        register = Avx2,
        op = generic_mul_vector,
        xconst = u8_xconst_avx2_nofma_mul_vector,
        xany = u8_xany_avx2_nofma_mul_vector,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = u8,
        register = Avx2,
        op = generic_div_vector,
        xconst = u8_xconst_avx2_nofma_div_vector,
        xany = u8_xany_avx2_nofma_div_vector,
        features = "avx2"
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = u16,
        register = Avx2,
        op = generic_add_value,
        xconst = u16_xconst_avx2_nofma_add_value,
        xany = u16_xany_avx2_nofma_add_value,
        features = "avx2"
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = u16,
        register = Avx2,
        op = generic_sub_value,
        xconst = u16_xconst_avx2_nofma_sub_value,
        xany = u16_xany_avx2_nofma_sub_value,
        features = "avx2"
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = u16,
        register = Avx2,
        op = generic_mul_value,
        xconst = u16_xconst_avx2_nofma_mul_value,
        xany = u16_xany_avx2_nofma_mul_value,
        features = "avx2"
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = u16,
        register = Avx2,
        op = generic_div_value,
        xconst = u16_xconst_avx2_nofma_div_value,
        xany = u16_xany_avx2_nofma_div_value,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = u16,
        register = Avx2,
        op = generic_add_vector,
        xconst = u16_xconst_avx2_nofma_add_vector,
        xany = u16_xany_avx2_nofma_add_vector,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = u16,
        register = Avx2,
        op = generic_sub_vector,
        xconst = u16_xconst_avx2_nofma_sub_vector,
        xany = u16_xany_avx2_nofma_sub_vector,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = u16,
        register = Avx2,
        op = generic_mul_vector,
        xconst = u16_xconst_avx2_nofma_mul_vector,
        xany = u16_xany_avx2_nofma_mul_vector,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = u16,
        register = Avx2,
        op = generic_div_vector,
        xconst = u16_xconst_avx2_nofma_div_vector,
        xany = u16_xany_avx2_nofma_div_vector,
        features = "avx2"
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = u32,
        register = Avx2,
        op = generic_add_value,
        xconst = u32_xconst_avx2_nofma_add_value,
        xany = u32_xany_avx2_nofma_add_value,
        features = "avx2"
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = u32,
        register = Avx2,
        op = generic_sub_value,
        xconst = u32_xconst_avx2_nofma_sub_value,
        xany = u32_xany_avx2_nofma_sub_value,
        features = "avx2"
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = u32,
        register = Avx2,
        op = generic_mul_value,
        xconst = u32_xconst_avx2_nofma_mul_value,
        xany = u32_xany_avx2_nofma_mul_value,
        features = "avx2"
    );
    export_vector_x_value_op!(
        description = "Value division, writing to a result vector",
        ty = u32,
        register = Avx2,
        op = generic_div_value,
        xconst = u32_xconst_avx2_nofma_div_value,
        xany = u32_xany_avx2_nofma_div_value,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = u32,
        register = Avx2,
        op = generic_add_vector,
        xconst = u32_xconst_avx2_nofma_add_vector,
        xany = u32_xany_avx2_nofma_add_vector,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = u32,
        register = Avx2,
        op = generic_sub_vector,
        xconst = u32_xconst_avx2_nofma_sub_vector,
        xany = u32_xany_avx2_nofma_sub_vector,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = u32,
        register = Avx2,
        op = generic_mul_vector,
        xconst = u32_xconst_avx2_nofma_mul_vector,
        xany = u32_xany_avx2_nofma_mul_vector,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = u32,
        register = Avx2,
        op = generic_div_vector,
        xconst = u32_xconst_avx2_nofma_div_vector,
        xany = u32_xany_avx2_nofma_div_vector,
        features = "avx2"
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = u64,
        register = Avx2,
        op = generic_add_value,
        xconst = u64_xconst_avx2_nofma_add_value,
        xany = u64_xany_avx2_nofma_add_value,
        features = "avx2"
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = u64,
        register = Avx2,
        op = generic_sub_value,
        xconst = u64_xconst_avx2_nofma_sub_value,
        xany = u64_xany_avx2_nofma_sub_value,
        features = "avx2"
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = u64,
        register = Avx2,
        op = generic_mul_value,
        xconst = u64_xconst_avx2_nofma_mul_value,
        xany = u64_xany_avx2_nofma_mul_value,
        features = "avx2"
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = u64,
        register = Avx2,
        op = generic_div_value,
        xconst = u64_xconst_avx2_nofma_div_value,
        xany = u64_xany_avx2_nofma_div_value,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = u64,
        register = Avx2,
        op = generic_add_vector,
        xconst = u64_xconst_avx2_nofma_add_vector,
        xany = u64_xany_avx2_nofma_add_vector,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = u64,
        register = Avx2,
        op = generic_sub_vector,
        xconst = u64_xconst_avx2_nofma_sub_vector,
        xany = u64_xany_avx2_nofma_sub_vector,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = u64,
        register = Avx2,
        op = generic_mul_vector,
        xconst = u64_xconst_avx2_nofma_mul_vector,
        xany = u64_xany_avx2_nofma_mul_vector,
        features = "avx2"
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = u64,
        register = Avx2,
        op = generic_div_vector,
        xconst = u64_xconst_avx2_nofma_div_vector,
        xany = u64_xany_avx2_nofma_div_vector,
        features = "avx2"
    );
}


#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
/// Vector space distance ops on the AVX512 supported architectures.
///
/// These methods **must** have AVX512f available to them at the time of calling
/// otherwise all of these methods become UB even if the input data is OK.
pub mod arithmetic_ops_avx512 {
    use super::*;

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = f32,
        register = Avx512,
        op = generic_add_value,
        xconst = f32_xconst_avx512_nofma_add_value,
        xany = f32_xany_avx512_nofma_add_value,
        features = "avx512f"
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = f32,
        register = Avx512,
        op = generic_sub_value,
        xconst = f32_xconst_avx512_nofma_sub_value,
        xany = f32_xany_avx512_nofma_sub_value,
        features = "avx512f"
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = f32,
        register = Avx512,
        op = generic_mul_value,
        xconst = f32_xconst_avx512_nofma_mul_value,
        xany = f32_xany_avx512_nofma_mul_value,
        features = "avx512f"
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = f32,
        register = Avx512,
        op = generic_div_value,
        xconst = f32_xconst_avx512_nofma_div_value,
        xany = f32_xany_avx512_nofma_div_value,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = f32,
        register = Avx512,
        op = generic_add_vector,
        xconst = f32_xconst_avx512_nofma_add_vector,
        xany = f32_xany_avx512_nofma_add_vector,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = f32,
        register = Avx512,
        op = generic_sub_vector,
        xconst = f32_xconst_avx512_nofma_sub_vector,
        xany = f32_xany_avx512_nofma_sub_vector,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector multiplication, writing to a result vector",
        ty = f32,
        register = Avx512,
        op = generic_mul_vector,
        xconst = f32_xconst_avx512_nofma_mul_vector,
        xany = f32_xany_avx512_nofma_mul_vector,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = f32,
        register = Avx512,
        op = generic_div_vector,
        xconst = f32_xconst_avx512_nofma_div_vector,
        xany = f32_xany_avx512_nofma_div_vector,
        features = "avx512f"
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = f64,
        register = Avx512,
        op = generic_add_value,
        xconst = f64_xconst_avx512_nofma_add_value,
        xany = f64_xany_avx512_nofma_add_value,
        features = "avx512f"
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = f64,
        register = Avx512,
        op = generic_sub_value,
        xconst = f64_xconst_avx512_nofma_sub_value,
        xany = f64_xany_avx512_nofma_sub_value,
        features = "avx512f"
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = f64,
        register = Avx512,
        op = generic_mul_value,
        xconst = f64_xconst_avx512_nofma_mul_value,
        xany = f64_xany_avx512_nofma_mul_value,
        features = "avx512f"
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = f64,
        register = Avx512,
        op = generic_div_value,
        xconst = f64_xconst_avx512_nofma_div_value,
        xany = f64_xany_avx512_nofma_div_value,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = f64,
        register = Avx512,
        op = generic_add_vector,
        xconst = f64_xconst_avx512_nofma_add_vector,
        xany = f64_xany_avx512_nofma_add_vector,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = f64,
        register = Avx512,
        op = generic_sub_vector,
        xconst = f64_xconst_avx512_nofma_sub_vector,
        xany = f64_xany_avx512_nofma_sub_vector,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = f64,
        register = Avx512,
        op = generic_mul_vector,
        xconst = f64_xconst_avx512_nofma_mul_vector,
        xany = f64_xany_avx512_nofma_mul_vector,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = f64,
        register = Avx512,
        op = generic_div_vector,
        xconst = f64_xconst_avx512_nofma_div_vector,
        xany = f64_xany_avx512_nofma_div_vector,
        features = "avx512f"
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = i8,
        register = Avx512,
        op = generic_add_value,
        xconst = i8_xconst_avx512_nofma_add_value,
        xany = i8_xany_avx512_nofma_add_value,
        features = "avx512f"
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = i8,
        register = Avx512,
        op = generic_sub_value,
        xconst = i8_xconst_avx512_nofma_sub_value,
        xany = i8_xany_avx512_nofma_sub_value,
        features = "avx512f"
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = i8,
        register = Avx512,
        op = generic_mul_value,
        xconst = i8_xconst_avx512_nofma_mul_value,
        xany = i8_xany_avx512_nofma_mul_value,
        features = "avx512f"
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = i8,
        register = Avx512,
        op = generic_div_value,
        xconst = i8_xconst_avx512_nofma_div_value,
        xany = i8_xany_avx512_nofma_div_value,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = i8,
        register = Avx512,
        op = generic_add_vector,
        xconst = i8_xconst_avx512_nofma_add_vector,
        xany = i8_xany_avx512_nofma_add_vector,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = i8,
        register = Avx512,
        op = generic_sub_vector,
        xconst = i8_xconst_avx512_nofma_sub_vector,
        xany = i8_xany_avx512_nofma_sub_vector,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = i8,
        register = Avx512,
        op = generic_mul_vector,
        xconst = i8_xconst_avx512_nofma_mul_vector,
        xany = i8_xany_avx512_nofma_mul_vector,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = i8,
        register = Avx512,
        op = generic_div_vector,
        xconst = i8_xconst_avx512_nofma_div_vector,
        xany = i8_xany_avx512_nofma_div_vector,
        features = "avx512f"
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = i16,
        register = Avx512,
        op = generic_add_value,
        xconst = i16_xconst_avx512_nofma_add_value,
        xany = i16_xany_avx512_nofma_add_value,
        features = "avx512f"
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = i16,
        register = Avx512,
        op = generic_sub_value,
        xconst = i16_xconst_avx512_nofma_sub_value,
        xany = i16_xany_avx512_nofma_sub_value,
        features = "avx512f"
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = i16,
        register = Avx512,
        op = generic_mul_value,
        xconst = i16_xconst_avx512_nofma_mul_value,
        xany = i16_xany_avx512_nofma_mul_value,
        features = "avx512f"
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = i16,
        register = Avx512,
        op = generic_div_value,
        xconst = i16_xconst_avx512_nofma_div_value,
        xany = i16_xany_avx512_nofma_div_value,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = i16,
        register = Avx512,
        op = generic_add_vector,
        xconst = i16_xconst_avx512_nofma_add_vector,
        xany = i16_xany_avx512_nofma_add_vector,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = i16,
        register = Avx512,
        op = generic_sub_vector,
        xconst = i16_xconst_avx512_nofma_sub_vector,
        xany = i16_xany_avx512_nofma_sub_vector,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = i16,
        register = Avx512,
        op = generic_mul_vector,
        xconst = i16_xconst_avx512_nofma_mul_vector,
        xany = i16_xany_avx512_nofma_mul_vector,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = i16,
        register = Avx512,
        op = generic_div_vector,
        xconst = i16_xconst_avx512_nofma_div_vector,
        xany = i16_xany_avx512_nofma_div_vector,
        features = "avx512f"
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = i32,
        register = Avx512,
        op = generic_add_value,
        xconst = i32_xconst_avx512_nofma_add_value,
        xany = i32_xany_avx512_nofma_add_value,
        features = "avx512f"
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = i32,
        register = Avx512,
        op = generic_sub_value,
        xconst = i32_xconst_avx512_nofma_sub_value,
        xany = i32_xany_avx512_nofma_sub_value,
        features = "avx512f"
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = i32,
        register = Avx512,
        op = generic_mul_value,
        xconst = i32_xconst_avx512_nofma_mul_value,
        xany = i32_xany_avx512_nofma_mul_value,
        features = "avx512f"
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = i32,
        register = Avx512,
        op = generic_div_value,
        xconst = i32_xconst_avx512_nofma_div_value,
        xany = i32_xany_avx512_nofma_div_value,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = i32,
        register = Avx512,
        op = generic_add_vector,
        xconst = i32_xconst_avx512_nofma_add_vector,
        xany = i32_xany_avx512_nofma_add_vector,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = i32,
        register = Avx512,
        op = generic_sub_vector,
        xconst = i32_xconst_avx512_nofma_sub_vector,
        xany = i32_xany_avx512_nofma_sub_vector,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = i32,
        register = Avx512,
        op = generic_mul_vector,
        xconst = i32_xconst_avx512_nofma_mul_vector,
        xany = i32_xany_avx512_nofma_mul_vector,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = i32,
        register = Avx512,
        op = generic_div_vector,
        xconst = i32_xconst_avx512_nofma_div_vector,
        xany = i32_xany_avx512_nofma_div_vector,
        features = "avx512f"
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = i64,
        register = Avx512,
        op = generic_add_value,
        xconst = i64_xconst_avx512_nofma_add_value,
        xany = i64_xany_avx512_nofma_add_value,
        features = "avx512f"
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = i64,
        register = Avx512,
        op = generic_sub_value,
        xconst = i64_xconst_avx512_nofma_sub_value,
        xany = i64_xany_avx512_nofma_sub_value,
        features = "avx512f"
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = i64,
        register = Avx512,
        op = generic_mul_value,
        xconst = i64_xconst_avx512_nofma_mul_value,
        xany = i64_xany_avx512_nofma_mul_value,
        features = "avx512f"
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = i64,
        register = Avx512,
        op = generic_div_value,
        xconst = i64_xconst_avx512_nofma_div_value,
        xany = i64_xany_avx512_nofma_div_value,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = i64,
        register = Avx512,
        op = generic_add_vector,
        xconst = i64_xconst_avx512_nofma_add_vector,
        xany = i64_xany_avx512_nofma_add_vector,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = i64,
        register = Avx512,
        op = generic_sub_vector,
        xconst = i64_xconst_avx512_nofma_sub_vector,
        xany = i64_xany_avx512_nofma_sub_vector,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = i64,
        register = Avx512,
        op = generic_mul_vector,
        xconst = i64_xconst_avx512_nofma_mul_vector,
        xany = i64_xany_avx512_nofma_mul_vector,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = i64,
        register = Avx512,
        op = generic_div_vector,
        xconst = i64_xconst_avx512_nofma_div_vector,
        xany = i64_xany_avx512_nofma_div_vector,
        features = "avx512f"
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = u8,
        register = Avx512,
        op = generic_add_value,
        xconst = u8_xconst_avx512_nofma_add_value,
        xany = u8_xany_avx512_nofma_add_value,
        features = "avx512f"
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = u8,
        register = Avx512,
        op = generic_sub_value,
        xconst = u8_xconst_avx512_nofma_sub_value,
        xany = u8_xany_avx512_nofma_sub_value,
        features = "avx512f"
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = u8,
        register = Avx512,
        op = generic_mul_value,
        xconst = u8_xconst_avx512_nofma_mul_value,
        xany = u8_xany_avx512_nofma_mul_value,
        features = "avx512f"
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = u8,
        register = Avx512,
        op = generic_div_value,
        xconst = u8_xconst_avx512_nofma_div_value,
        xany = u8_xany_avx512_nofma_div_value,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = u8,
        register = Avx512,
        op = generic_add_vector,
        xconst = u8_xconst_avx512_nofma_add_vector,
        xany = u8_xany_avx512_nofma_add_vector,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = u8,
        register = Avx512,
        op = generic_sub_vector,
        xconst = u8_xconst_avx512_nofma_sub_vector,
        xany = u8_xany_avx512_nofma_sub_vector,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = u8,
        register = Avx512,
        op = generic_mul_vector,
        xconst = u8_xconst_avx512_nofma_mul_vector,
        xany = u8_xany_avx512_nofma_mul_vector,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = u8,
        register = Avx512,
        op = generic_div_vector,
        xconst = u8_xconst_avx512_nofma_div_vector,
        xany = u8_xany_avx512_nofma_div_vector,
        features = "avx512f"
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = u16,
        register = Avx512,
        op = generic_add_value,
        xconst = u16_xconst_avx512_nofma_add_value,
        xany = u16_xany_avx512_nofma_add_value,
        features = "avx512f"
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = u16,
        register = Avx512,
        op = generic_sub_value,
        xconst = u16_xconst_avx512_nofma_sub_value,
        xany = u16_xany_avx512_nofma_sub_value,
        features = "avx512f"
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = u16,
        register = Avx512,
        op = generic_mul_value,
        xconst = u16_xconst_avx512_nofma_mul_value,
        xany = u16_xany_avx512_nofma_mul_value,
        features = "avx512f"
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = u16,
        register = Avx512,
        op = generic_div_value,
        xconst = u16_xconst_avx512_nofma_div_value,
        xany = u16_xany_avx512_nofma_div_value,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = u16,
        register = Avx512,
        op = generic_add_vector,
        xconst = u16_xconst_avx512_nofma_add_vector,
        xany = u16_xany_avx512_nofma_add_vector,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = u16,
        register = Avx512,
        op = generic_sub_vector,
        xconst = u16_xconst_avx512_nofma_sub_vector,
        xany = u16_xany_avx512_nofma_sub_vector,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = u16,
        register = Avx512,
        op = generic_mul_vector,
        xconst = u16_xconst_avx512_nofma_mul_vector,
        xany = u16_xany_avx512_nofma_mul_vector,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = u16,
        register = Avx512,
        op = generic_div_vector,
        xconst = u16_xconst_avx512_nofma_div_vector,
        xany = u16_xany_avx512_nofma_div_vector,
        features = "avx512f"
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = u32,
        register = Avx512,
        op = generic_add_value,
        xconst = u32_xconst_avx512_nofma_add_value,
        xany = u32_xany_avx512_nofma_add_value,
        features = "avx512f"
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = u32,
        register = Avx512,
        op = generic_sub_value,
        xconst = u32_xconst_avx512_nofma_sub_value,
        xany = u32_xany_avx512_nofma_sub_value,
        features = "avx512f"
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = u32,
        register = Avx512,
        op = generic_mul_value,
        xconst = u32_xconst_avx512_nofma_mul_value,
        xany = u32_xany_avx512_nofma_mul_value,
        features = "avx512f"
    );
    export_vector_x_value_op!(
        description = "Value division, writing to a result vector",
        ty = u32,
        register = Avx512,
        op = generic_div_value,
        xconst = u32_xconst_avx512_nofma_div_value,
        xany = u32_xany_avx512_nofma_div_value,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = u32,
        register = Avx512,
        op = generic_add_vector,
        xconst = u32_xconst_avx512_nofma_add_vector,
        xany = u32_xany_avx512_nofma_add_vector,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = u32,
        register = Avx512,
        op = generic_sub_vector,
        xconst = u32_xconst_avx512_nofma_sub_vector,
        xany = u32_xany_avx512_nofma_sub_vector,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = u32,
        register = Avx512,
        op = generic_mul_vector,
        xconst = u32_xconst_avx512_nofma_mul_vector,
        xany = u32_xany_avx512_nofma_mul_vector,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = u32,
        register = Avx512,
        op = generic_div_vector,
        xconst = u32_xconst_avx512_nofma_div_vector,
        xany = u32_xany_avx512_nofma_div_vector,
        features = "avx512f"
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = u64,
        register = Avx512,
        op = generic_add_value,
        xconst = u64_xconst_avx512_nofma_add_value,
        xany = u64_xany_avx512_nofma_add_value,
        features = "avx512f"
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = u64,
        register = Avx512,
        op = generic_sub_value,
        xconst = u64_xconst_avx512_nofma_sub_value,
        xany = u64_xany_avx512_nofma_sub_value,
        features = "avx512f"
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = u64,
        register = Avx512,
        op = generic_mul_value,
        xconst = u64_xconst_avx512_nofma_mul_value,
        xany = u64_xany_avx512_nofma_mul_value,
        features = "avx512f"
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = u64,
        register = Avx512,
        op = generic_div_value,
        xconst = u64_xconst_avx512_nofma_div_value,
        xany = u64_xany_avx512_nofma_div_value,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = u64,
        register = Avx512,
        op = generic_add_vector,
        xconst = u64_xconst_avx512_nofma_add_vector,
        xany = u64_xany_avx512_nofma_add_vector,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = u64,
        register = Avx512,
        op = generic_sub_vector,
        xconst = u64_xconst_avx512_nofma_sub_vector,
        xany = u64_xany_avx512_nofma_sub_vector,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = u64,
        register = Avx512,
        op = generic_mul_vector,
        xconst = u64_xconst_avx512_nofma_mul_vector,
        xany = u64_xany_avx512_nofma_mul_vector,
        features = "avx512f"
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = u64,
        register = Avx512,
        op = generic_div_vector,
        xconst = u64_xconst_avx512_nofma_div_vector,
        xany = u64_xany_avx512_nofma_div_vector,
        features = "avx512f"
    );
}

#[cfg(target_arch = "aarch64")]
/// Vector space distance ops on the NEON supported architectures.
///
/// These methods **must** have NEON available to them at the time of calling
/// otherwise all of these methods become UB even if the input data is OK.
pub mod arithmetic_ops_neon {
    use super::*;

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = f32,
        register = Neon,
        op = generic_add_value,
        xconst = f32_xconst_neon_nofma_add_value,
        xany = f32_xany_neon_nofma_add_value,
        features = "neon"
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = f32,
        register = Neon,
        op = generic_sub_value,
        xconst = f32_xconst_neon_nofma_sub_value,
        xany = f32_xany_neon_nofma_sub_value,
        features = "neon"
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = f32,
        register = Neon,
        op = generic_mul_value,
        xconst = f32_xconst_neon_nofma_mul_value,
        xany = f32_xany_neon_nofma_mul_value,
        features = "neon"
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = f32,
        register = Neon,
        op = generic_div_value,
        xconst = f32_xconst_neon_nofma_div_value,
        xany = f32_xany_neon_nofma_div_value,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = f32,
        register = Neon,
        op = generic_add_vector,
        xconst = f32_xconst_neon_nofma_add_vector,
        xany = f32_xany_neon_nofma_add_vector,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = f32,
        register = Neon,
        op = generic_sub_vector,
        xconst = f32_xconst_neon_nofma_sub_vector,
        xany = f32_xany_neon_nofma_sub_vector,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector multiplication, writing to a result vector",
        ty = f32,
        register = Neon,
        op = generic_mul_vector,
        xconst = f32_xconst_neon_nofma_mul_vector,
        xany = f32_xany_neon_nofma_mul_vector,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = f32,
        register = Neon,
        op = generic_div_vector,
        xconst = f32_xconst_neon_nofma_div_vector,
        xany = f32_xany_neon_nofma_div_vector,
        features = "neon"
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = f64,
        register = Neon,
        op = generic_add_value,
        xconst = f64_xconst_neon_nofma_add_value,
        xany = f64_xany_neon_nofma_add_value,
        features = "neon"
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = f64,
        register = Neon,
        op = generic_sub_value,
        xconst = f64_xconst_neon_nofma_sub_value,
        xany = f64_xany_neon_nofma_sub_value,
        features = "neon"
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = f64,
        register = Neon,
        op = generic_mul_value,
        xconst = f64_xconst_neon_nofma_mul_value,
        xany = f64_xany_neon_nofma_mul_value,
        features = "neon"
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = f64,
        register = Neon,
        op = generic_div_value,
        xconst = f64_xconst_neon_nofma_div_value,
        xany = f64_xany_neon_nofma_div_value,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = f64,
        register = Neon,
        op = generic_add_vector,
        xconst = f64_xconst_neon_nofma_add_vector,
        xany = f64_xany_neon_nofma_add_vector,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = f64,
        register = Neon,
        op = generic_sub_vector,
        xconst = f64_xconst_neon_nofma_sub_vector,
        xany = f64_xany_neon_nofma_sub_vector,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = f64,
        register = Neon,
        op = generic_mul_vector,
        xconst = f64_xconst_neon_nofma_mul_vector,
        xany = f64_xany_neon_nofma_mul_vector,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = f64,
        register = Neon,
        op = generic_div_vector,
        xconst = f64_xconst_neon_nofma_div_vector,
        xany = f64_xany_neon_nofma_div_vector,
        features = "neon"
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = i8,
        register = Neon,
        op = generic_add_value,
        xconst = i8_xconst_neon_nofma_add_value,
        xany = i8_xany_neon_nofma_add_value,
        features = "neon"
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = i8,
        register = Neon,
        op = generic_sub_value,
        xconst = i8_xconst_neon_nofma_sub_value,
        xany = i8_xany_neon_nofma_sub_value,
        features = "neon"
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = i8,
        register = Neon,
        op = generic_mul_value,
        xconst = i8_xconst_neon_nofma_mul_value,
        xany = i8_xany_neon_nofma_mul_value,
        features = "neon"
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = i8,
        register = Neon,
        op = generic_div_value,
        xconst = i8_xconst_neon_nofma_div_value,
        xany = i8_xany_neon_nofma_div_value,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = i8,
        register = Neon,
        op = generic_add_vector,
        xconst = i8_xconst_neon_nofma_add_vector,
        xany = i8_xany_neon_nofma_add_vector,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = i8,
        register = Neon,
        op = generic_sub_vector,
        xconst = i8_xconst_neon_nofma_sub_vector,
        xany = i8_xany_neon_nofma_sub_vector,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = i8,
        register = Neon,
        op = generic_mul_vector,
        xconst = i8_xconst_neon_nofma_mul_vector,
        xany = i8_xany_neon_nofma_mul_vector,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = i8,
        register = Neon,
        op = generic_div_vector,
        xconst = i8_xconst_neon_nofma_div_vector,
        xany = i8_xany_neon_nofma_div_vector,
        features = "neon"
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = i16,
        register = Neon,
        op = generic_add_value,
        xconst = i16_xconst_neon_nofma_add_value,
        xany = i16_xany_neon_nofma_add_value,
        features = "neon"
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = i16,
        register = Neon,
        op = generic_sub_value,
        xconst = i16_xconst_neon_nofma_sub_value,
        xany = i16_xany_neon_nofma_sub_value,
        features = "neon"
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = i16,
        register = Neon,
        op = generic_mul_value,
        xconst = i16_xconst_neon_nofma_mul_value,
        xany = i16_xany_neon_nofma_mul_value,
        features = "neon"
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = i16,
        register = Neon,
        op = generic_div_value,
        xconst = i16_xconst_neon_nofma_div_value,
        xany = i16_xany_neon_nofma_div_value,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = i16,
        register = Neon,
        op = generic_add_vector,
        xconst = i16_xconst_neon_nofma_add_vector,
        xany = i16_xany_neon_nofma_add_vector,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = i16,
        register = Neon,
        op = generic_sub_vector,
        xconst = i16_xconst_neon_nofma_sub_vector,
        xany = i16_xany_neon_nofma_sub_vector,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = i16,
        register = Neon,
        op = generic_mul_vector,
        xconst = i16_xconst_neon_nofma_mul_vector,
        xany = i16_xany_neon_nofma_mul_vector,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = i16,
        register = Neon,
        op = generic_div_vector,
        xconst = i16_xconst_neon_nofma_div_vector,
        xany = i16_xany_neon_nofma_div_vector,
        features = "neon"
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = i32,
        register = Neon,
        op = generic_add_value,
        xconst = i32_xconst_neon_nofma_add_value,
        xany = i32_xany_neon_nofma_add_value,
        features = "neon"
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = i32,
        register = Neon,
        op = generic_sub_value,
        xconst = i32_xconst_neon_nofma_sub_value,
        xany = i32_xany_neon_nofma_sub_value,
        features = "neon"
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = i32,
        register = Neon,
        op = generic_mul_value,
        xconst = i32_xconst_neon_nofma_mul_value,
        xany = i32_xany_neon_nofma_mul_value,
        features = "neon"
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = i32,
        register = Neon,
        op = generic_div_value,
        xconst = i32_xconst_neon_nofma_div_value,
        xany = i32_xany_neon_nofma_div_value,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = i32,
        register = Neon,
        op = generic_add_vector,
        xconst = i32_xconst_neon_nofma_add_vector,
        xany = i32_xany_neon_nofma_add_vector,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = i32,
        register = Neon,
        op = generic_sub_vector,
        xconst = i32_xconst_neon_nofma_sub_vector,
        xany = i32_xany_neon_nofma_sub_vector,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = i32,
        register = Neon,
        op = generic_mul_vector,
        xconst = i32_xconst_neon_nofma_mul_vector,
        xany = i32_xany_neon_nofma_mul_vector,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = i32,
        register = Neon,
        op = generic_div_vector,
        xconst = i32_xconst_neon_nofma_div_vector,
        xany = i32_xany_neon_nofma_div_vector,
        features = "neon"
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = i64,
        register = Neon,
        op = generic_add_value,
        xconst = i64_xconst_neon_nofma_add_value,
        xany = i64_xany_neon_nofma_add_value,
        features = "neon"
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = i64,
        register = Neon,
        op = generic_sub_value,
        xconst = i64_xconst_neon_nofma_sub_value,
        xany = i64_xany_neon_nofma_sub_value,
        features = "neon"
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = i64,
        register = Neon,
        op = generic_mul_value,
        xconst = i64_xconst_neon_nofma_mul_value,
        xany = i64_xany_neon_nofma_mul_value,
        features = "neon"
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = i64,
        register = Neon,
        op = generic_div_value,
        xconst = i64_xconst_neon_nofma_div_value,
        xany = i64_xany_neon_nofma_div_value,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = i64,
        register = Neon,
        op = generic_add_vector,
        xconst = i64_xconst_neon_nofma_add_vector,
        xany = i64_xany_neon_nofma_add_vector,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = i64,
        register = Neon,
        op = generic_sub_vector,
        xconst = i64_xconst_neon_nofma_sub_vector,
        xany = i64_xany_neon_nofma_sub_vector,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = i64,
        register = Neon,
        op = generic_mul_vector,
        xconst = i64_xconst_neon_nofma_mul_vector,
        xany = i64_xany_neon_nofma_mul_vector,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = i64,
        register = Neon,
        op = generic_div_vector,
        xconst = i64_xconst_neon_nofma_div_vector,
        xany = i64_xany_neon_nofma_div_vector,
        features = "neon"
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = u8,
        register = Neon,
        op = generic_add_value,
        xconst = u8_xconst_neon_nofma_add_value,
        xany = u8_xany_neon_nofma_add_value,
        features = "neon"
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = u8,
        register = Neon,
        op = generic_sub_value,
        xconst = u8_xconst_neon_nofma_sub_value,
        xany = u8_xany_neon_nofma_sub_value,
        features = "neon"
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = u8,
        register = Neon,
        op = generic_mul_value,
        xconst = u8_xconst_neon_nofma_mul_value,
        xany = u8_xany_neon_nofma_mul_value,
        features = "neon"
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = u8,
        register = Neon,
        op = generic_div_value,
        xconst = u8_xconst_neon_nofma_div_value,
        xany = u8_xany_neon_nofma_div_value,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = u8,
        register = Neon,
        op = generic_add_vector,
        xconst = u8_xconst_neon_nofma_add_vector,
        xany = u8_xany_neon_nofma_add_vector,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = u8,
        register = Neon,
        op = generic_sub_vector,
        xconst = u8_xconst_neon_nofma_sub_vector,
        xany = u8_xany_neon_nofma_sub_vector,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = u8,
        register = Neon,
        op = generic_mul_vector,
        xconst = u8_xconst_neon_nofma_mul_vector,
        xany = u8_xany_neon_nofma_mul_vector,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = u8,
        register = Neon,
        op = generic_div_vector,
        xconst = u8_xconst_neon_nofma_div_vector,
        xany = u8_xany_neon_nofma_div_vector,
        features = "neon"
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = u16,
        register = Neon,
        op = generic_add_value,
        xconst = u16_xconst_neon_nofma_add_value,
        xany = u16_xany_neon_nofma_add_value,
        features = "neon"
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = u16,
        register = Neon,
        op = generic_sub_value,
        xconst = u16_xconst_neon_nofma_sub_value,
        xany = u16_xany_neon_nofma_sub_value,
        features = "neon"
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = u16,
        register = Neon,
        op = generic_mul_value,
        xconst = u16_xconst_neon_nofma_mul_value,
        xany = u16_xany_neon_nofma_mul_value,
        features = "neon"
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = u16,
        register = Neon,
        op = generic_div_value,
        xconst = u16_xconst_neon_nofma_div_value,
        xany = u16_xany_neon_nofma_div_value,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = u16,
        register = Neon,
        op = generic_add_vector,
        xconst = u16_xconst_neon_nofma_add_vector,
        xany = u16_xany_neon_nofma_add_vector,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = u16,
        register = Neon,
        op = generic_sub_vector,
        xconst = u16_xconst_neon_nofma_sub_vector,
        xany = u16_xany_neon_nofma_sub_vector,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = u16,
        register = Neon,
        op = generic_mul_vector,
        xconst = u16_xconst_neon_nofma_mul_vector,
        xany = u16_xany_neon_nofma_mul_vector,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = u16,
        register = Neon,
        op = generic_div_vector,
        xconst = u16_xconst_neon_nofma_div_vector,
        xany = u16_xany_neon_nofma_div_vector,
        features = "neon"
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = u32,
        register = Neon,
        op = generic_add_value,
        xconst = u32_xconst_neon_nofma_add_value,
        xany = u32_xany_neon_nofma_add_value,
        features = "neon"
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = u32,
        register = Neon,
        op = generic_sub_value,
        xconst = u32_xconst_neon_nofma_sub_value,
        xany = u32_xany_neon_nofma_sub_value,
        features = "neon"
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = u32,
        register = Neon,
        op = generic_mul_value,
        xconst = u32_xconst_neon_nofma_mul_value,
        xany = u32_xany_neon_nofma_mul_value,
        features = "neon"
    );
    export_vector_x_value_op!(
        description = "Value division, writing to a result vector",
        ty = u32,
        register = Neon,
        op = generic_div_value,
        xconst = u32_xconst_neon_nofma_div_value,
        xany = u32_xany_neon_nofma_div_value,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = u32,
        register = Neon,
        op = generic_add_vector,
        xconst = u32_xconst_neon_nofma_add_vector,
        xany = u32_xany_neon_nofma_add_vector,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = u32,
        register = Neon,
        op = generic_sub_vector,
        xconst = u32_xconst_neon_nofma_sub_vector,
        xany = u32_xany_neon_nofma_sub_vector,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = u32,
        register = Neon,
        op = generic_mul_vector,
        xconst = u32_xconst_neon_nofma_mul_vector,
        xany = u32_xany_neon_nofma_mul_vector,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = u32,
        register = Neon,
        op = generic_div_vector,
        xconst = u32_xconst_neon_nofma_div_vector,
        xany = u32_xany_neon_nofma_div_vector,
        features = "neon"
    );

    export_vector_x_value_op!(
        description = "Value addition on a provided vector, writing to a result vector",
        ty = u64,
        register = Neon,
        op = generic_add_value,
        xconst = u64_xconst_neon_nofma_add_value,
        xany = u64_xany_neon_nofma_add_value,
        features = "neon"
    );
    export_vector_x_value_op!(
        description = "Value subtraction on a provided vector, writing to a result vector",
        ty = u64,
        register = Neon,
        op = generic_sub_value,
        xconst = u64_xconst_neon_nofma_sub_value,
        xany = u64_xany_neon_nofma_sub_value,
        features = "neon"
    );
    export_vector_x_value_op!(
        description = "Value multiply on a provided vector, writing to a result vector",
        ty = u64,
        register = Neon,
        op = generic_mul_value,
        xconst = u64_xconst_neon_nofma_mul_value,
        xany = u64_xany_neon_nofma_mul_value,
        features = "neon"
    );
    export_vector_x_value_op!(
        description = "Value division on a provided vector, writing to a result vector",
        ty = u64,
        register = Neon,
        op = generic_div_value,
        xconst = u64_xconst_neon_nofma_div_value,
        xany = u64_xany_neon_nofma_div_value,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector addition, writing to a result vector",
        ty = u64,
        register = Neon,
        op = generic_add_vector,
        xconst = u64_xconst_neon_nofma_add_vector,
        xany = u64_xany_neon_nofma_add_vector,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector subtraction, writing to a result vector",
        ty = u64,
        register = Neon,
        op = generic_sub_vector,
        xconst = u64_xconst_neon_nofma_sub_vector,
        xany = u64_xany_neon_nofma_sub_vector,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector multiply, writing to a result vector",
        ty = u64,
        register = Neon,
        op = generic_mul_vector,
        xconst = u64_xconst_neon_nofma_mul_vector,
        xany = u64_xany_neon_nofma_mul_vector,
        features = "neon"
    );
    export_vector_x_vector_op!(
        description = "Vector division, writing to a result vector",
        ty = u64,
        register = Neon,
        op = generic_div_vector,
        xconst = u64_xconst_neon_nofma_div_vector,
        xany = u64_xany_neon_nofma_div_vector,
        features = "neon"
    );
}
