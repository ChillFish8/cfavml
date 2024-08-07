//! Generated non-generic methods for SIMD based min/max/sum vector operations.
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

macro_rules! export_op_horizontal {
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
        pub unsafe fn $xconst_name<const DIMS: usize>(a: &[$t]) -> $t {
            $op::<_, $im, AutoMath>(DIMS, a)
        }

        #[doc = concat!("`", stringify!($t), "` ", $desc)]
        ///
        /// # Safety
        ///
        /// Vectors **`a`** and **`result`** must be equal in length otherwise out of bound
        /// access can occur.
        pub unsafe fn $xany_name(a: &[$t]) -> $t {
            $op::<_, $im, AutoMath>(a.len(), a)
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
        pub unsafe fn $xconst_name<const DIMS: usize>(a: &[$t]) -> $t {
            $op::<_, $im, AutoMath>(DIMS, a)
        }

        #[target_feature($(enable = $feat , )*)]
        #[doc = concat!("`", stringify!($t), "` ", $desc)]
        #[doc = "\n\n# Safety\n\n"]
        #[doc = concat!("The following CPU flags must be available: ", $("**", $feat, "**", ", ",)*)]
        ///
        /// Vectors **`a`** and **`result`** must be equal in length otherwise out of bound
        /// access can occur.
        pub unsafe fn $xany_name(a: &[$t]) -> $t {
            $op::<_, $im, AutoMath>(a.len(), a)
        }
    };
}

macro_rules! export_op_vertical {
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
        /// Vectors **`a`** and **`result`** must be equal in length otherwise out of bound
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

macro_rules! export_op_value {
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
        /// Vectors **`a`**, **`b`** and **`result`** must be equal in length to that specified in **`DIMS`**,
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
        /// Vectors **`a`**, **`b`** and **`result`** must be equal in length otherwise out of bound
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

/// Vector min/max/sum ops on the fallback implementations.
///
/// These methods do not strictly require any CPU feature but do auto-vectorize
/// well when target features are explicitly provided and can be used to
/// provide accelerated operations on non-explicitly supported SIMD architectures.
pub mod min_max_sum_ops_fallback {
    use super::*;

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = f32,
        register = Fallback,
        op = generic_squared_norm,
        xconst = f32_xconst_fallback_nofma_squared_norm,
        xany = f32_xany_fallback_nofma_squared_norm,
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = f32,
        register = Fallback,
        op = generic_sum,
        xconst = f32_xconst_fallback_nofma_sum,
        xany = f32_xany_fallback_nofma_sum,
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = f32,
        register = Fallback,
        op = generic_max_horizontal,
        xconst = f32_xconst_fallback_nofma_max_horizontal,
        xany = f32_xany_fallback_nofma_max_horizontal,
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = f32,
        register = Fallback,
        op = generic_min_horizontal,
        xconst = f32_xconst_fallback_nofma_min_horizontal,
        xany = f32_xany_fallback_nofma_min_horizontal,
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = f32,
        register = Fallback,
        op = generic_max_vertical,
        xconst = f32_xconst_fallback_nofma_max_vertical,
        xany = f32_xany_fallback_nofma_max_vertical,
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = f32,
        register = Fallback,
        op = generic_min_vertical,
        xconst = f32_xconst_fallback_nofma_min_vertical,
        xany = f32_xany_fallback_nofma_min_vertical,
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = f32,
        register = Fallback,
        op = generic_max_value,
        xconst = f32_xconst_fallback_nofma_max_value,
        xany = f32_xany_fallback_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = f32,
        register = Fallback,
        op = generic_min_value,
        xconst = f32_xconst_fallback_nofma_min_value,
        xany = f32_xany_fallback_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = f64,
        register = Fallback,
        op = generic_squared_norm,
        xconst = f64_xconst_fallback_nofma_squared_norm,
        xany = f64_xany_fallback_nofma_squared_norm,
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = f64,
        register = Fallback,
        op = generic_sum,
        xconst = f64_xconst_fallback_nofma_sum,
        xany = f64_xany_fallback_nofma_sum,
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = f64,
        register = Fallback,
        op = generic_max_horizontal,
        xconst = f64_xconst_fallback_nofma_max_horizontal,
        xany = f64_xany_fallback_nofma_max_horizontal,
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = f64,
        register = Fallback,
        op = generic_min_horizontal,
        xconst = f64_xconst_fallback_nofma_min_horizontal,
        xany = f64_xany_fallback_nofma_min_horizontal,
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = f64,
        register = Fallback,
        op = generic_max_vertical,
        xconst = f64_xconst_fallback_nofma_max_vertical,
        xany = f64_xany_fallback_nofma_max_vertical,
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = f64,
        register = Fallback,
        op = generic_min_vertical,
        xconst = f64_xconst_fallback_nofma_min_vertical,
        xany = f64_xany_fallback_nofma_min_vertical,
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = f64,
        register = Fallback,
        op = generic_max_value,
        xconst = f64_xconst_fallback_nofma_max_value,
        xany = f64_xany_fallback_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = f64,
        register = Fallback,
        op = generic_min_value,
        xconst = f64_xconst_fallback_nofma_min_value,
        xany = f64_xany_fallback_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = u8,
        register = Fallback,
        op = generic_squared_norm,
        xconst = u8_xconst_fallback_nofma_squared_norm,
        xany = u8_xany_fallback_nofma_squared_norm,
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = u8,
        register = Fallback,
        op = generic_sum,
        xconst = u8_xconst_fallback_nofma_sum,
        xany = u8_xany_fallback_nofma_sum,
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = u8,
        register = Fallback,
        op = generic_max_horizontal,
        xconst = u8_xconst_fallback_nofma_max_horizontal,
        xany = u8_xany_fallback_nofma_max_horizontal,
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = u8,
        register = Fallback,
        op = generic_min_horizontal,
        xconst = u8_xconst_fallback_nofma_min_horizontal,
        xany = u8_xany_fallback_nofma_min_horizontal,
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = u8,
        register = Fallback,
        op = generic_max_vertical,
        xconst = u8_xconst_fallback_nofma_max_vertical,
        xany = u8_xany_fallback_nofma_max_vertical,
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = u8,
        register = Fallback,
        op = generic_min_vertical,
        xconst = u8_xconst_fallback_nofma_min_vertical,
        xany = u8_xany_fallback_nofma_min_vertical,
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = u8,
        register = Fallback,
        op = generic_max_value,
        xconst = u8_xconst_fallback_nofma_max_value,
        xany = u8_xany_fallback_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = u8,
        register = Fallback,
        op = generic_min_value,
        xconst = u8_xconst_fallback_nofma_min_value,
        xany = u8_xany_fallback_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = u16,
        register = Fallback,
        op = generic_squared_norm,
        xconst = u16_xconst_fallback_nofma_squared_norm,
        xany = u16_xany_fallback_nofma_squared_norm,
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = u16,
        register = Fallback,
        op = generic_sum,
        xconst = u16_xconst_fallback_nofma_sum,
        xany = u16_xany_fallback_nofma_sum,
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = u16,
        register = Fallback,
        op = generic_max_horizontal,
        xconst = u16_xconst_fallback_nofma_max_horizontal,
        xany = u16_xany_fallback_nofma_max_horizontal,
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = u16,
        register = Fallback,
        op = generic_min_horizontal,
        xconst = u16_xconst_fallback_nofma_min_horizontal,
        xany = u16_xany_fallback_nofma_min_horizontal,
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = u16,
        register = Fallback,
        op = generic_max_vertical,
        xconst = u16_xconst_fallback_nofma_max_vertical,
        xany = u16_xany_fallback_nofma_max_vertical,
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = u16,
        register = Fallback,
        op = generic_min_vertical,
        xconst = u16_xconst_fallback_nofma_min_vertical,
        xany = u16_xany_fallback_nofma_min_vertical,
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = u16,
        register = Fallback,
        op = generic_max_value,
        xconst = u16_xconst_fallback_nofma_max_value,
        xany = u16_xany_fallback_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = u16,
        register = Fallback,
        op = generic_min_value,
        xconst = u16_xconst_fallback_nofma_min_value,
        xany = u16_xany_fallback_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = u32,
        register = Fallback,
        op = generic_squared_norm,
        xconst = u32_xconst_fallback_nofma_squared_norm,
        xany = u32_xany_fallback_nofma_squared_norm,
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = u32,
        register = Fallback,
        op = generic_sum,
        xconst = u32_xconst_fallback_nofma_sum,
        xany = u32_xany_fallback_nofma_sum,
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = u32,
        register = Fallback,
        op = generic_max_horizontal,
        xconst = u32_xconst_fallback_nofma_max_horizontal,
        xany = u32_xany_fallback_nofma_max_horizontal,
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = u32,
        register = Fallback,
        op = generic_min_horizontal,
        xconst = u32_xconst_fallback_nofma_min_horizontal,
        xany = u32_xany_fallback_nofma_min_horizontal,
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = u32,
        register = Fallback,
        op = generic_max_vertical,
        xconst = u32_xconst_fallback_nofma_max_vertical,
        xany = u32_xany_fallback_nofma_max_vertical,
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = u32,
        register = Fallback,
        op = generic_min_vertical,
        xconst = u32_xconst_fallback_nofma_min_vertical,
        xany = u32_xany_fallback_nofma_min_vertical,
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = u32,
        register = Fallback,
        op = generic_max_value,
        xconst = u32_xconst_fallback_nofma_max_value,
        xany = u32_xany_fallback_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = u32,
        register = Fallback,
        op = generic_min_value,
        xconst = u32_xconst_fallback_nofma_min_value,
        xany = u32_xany_fallback_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = u64,
        register = Fallback,
        op = generic_squared_norm,
        xconst = u64_xconst_fallback_nofma_squared_norm,
        xany = u64_xany_fallback_nofma_squared_norm,
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = u64,
        register = Fallback,
        op = generic_sum,
        xconst = u64_xconst_fallback_nofma_sum,
        xany = u64_xany_fallback_nofma_sum,
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = u64,
        register = Fallback,
        op = generic_max_horizontal,
        xconst = u64_xconst_fallback_nofma_max_horizontal,
        xany = u64_xany_fallback_nofma_max_horizontal,
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = u64,
        register = Fallback,
        op = generic_min_horizontal,
        xconst = u64_xconst_fallback_nofma_min_horizontal,
        xany = u64_xany_fallback_nofma_min_horizontal,
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = u64,
        register = Fallback,
        op = generic_max_vertical,
        xconst = u64_xconst_fallback_nofma_max_vertical,
        xany = u64_xany_fallback_nofma_max_vertical,
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = u64,
        register = Fallback,
        op = generic_min_vertical,
        xconst = u64_xconst_fallback_nofma_min_vertical,
        xany = u64_xany_fallback_nofma_min_vertical,
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = u64,
        register = Fallback,
        op = generic_max_value,
        xconst = u64_xconst_fallback_nofma_max_value,
        xany = u64_xany_fallback_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = u64,
        register = Fallback,
        op = generic_min_value,
        xconst = u64_xconst_fallback_nofma_min_value,
        xany = u64_xany_fallback_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = i8,
        register = Fallback,
        op = generic_squared_norm,
        xconst = i8_xconst_fallback_nofma_squared_norm,
        xany = i8_xany_fallback_nofma_squared_norm,
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = i8,
        register = Fallback,
        op = generic_sum,
        xconst = i8_xconst_fallback_nofma_sum,
        xany = i8_xany_fallback_nofma_sum,
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = i8,
        register = Fallback,
        op = generic_max_horizontal,
        xconst = i8_xconst_fallback_nofma_max_horizontal,
        xany = i8_xany_fallback_nofma_max_horizontal,
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = i8,
        register = Fallback,
        op = generic_min_horizontal,
        xconst = i8_xconst_fallback_nofma_min_horizontal,
        xany = i8_xany_fallback_nofma_min_horizontal,
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = i8,
        register = Fallback,
        op = generic_max_vertical,
        xconst = i8_xconst_fallback_nofma_max_vertical,
        xany = i8_xany_fallback_nofma_max_vertical,
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = i8,
        register = Fallback,
        op = generic_min_vertical,
        xconst = i8_xconst_fallback_nofma_min_vertical,
        xany = i8_xany_fallback_nofma_min_vertical,
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = i8,
        register = Fallback,
        op = generic_max_value,
        xconst = i8_xconst_fallback_nofma_max_value,
        xany = i8_xany_fallback_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = i8,
        register = Fallback,
        op = generic_min_value,
        xconst = i8_xconst_fallback_nofma_min_value,
        xany = i8_xany_fallback_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = i16,
        register = Fallback,
        op = generic_squared_norm,
        xconst = i16_xconst_fallback_nofma_squared_norm,
        xany = i16_xany_fallback_nofma_squared_norm,
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = i16,
        register = Fallback,
        op = generic_sum,
        xconst = i16_xconst_fallback_nofma_sum,
        xany = i16_xany_fallback_nofma_sum,
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = i16,
        register = Fallback,
        op = generic_max_horizontal,
        xconst = i16_xconst_fallback_nofma_max_horizontal,
        xany = i16_xany_fallback_nofma_max_horizontal,
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = i16,
        register = Fallback,
        op = generic_min_horizontal,
        xconst = i16_xconst_fallback_nofma_min_horizontal,
        xany = i16_xany_fallback_nofma_min_horizontal,
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = i16,
        register = Fallback,
        op = generic_max_vertical,
        xconst = i16_xconst_fallback_nofma_max_vertical,
        xany = i16_xany_fallback_nofma_max_vertical,
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = i16,
        register = Fallback,
        op = generic_min_vertical,
        xconst = i16_xconst_fallback_nofma_min_vertical,
        xany = i16_xany_fallback_nofma_min_vertical,
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = i16,
        register = Fallback,
        op = generic_max_value,
        xconst = i16_xconst_fallback_nofma_max_value,
        xany = i16_xany_fallback_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = i16,
        register = Fallback,
        op = generic_min_value,
        xconst = i16_xconst_fallback_nofma_min_value,
        xany = i16_xany_fallback_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = i32,
        register = Fallback,
        op = generic_squared_norm,
        xconst = i32_xconst_fallback_nofma_squared_norm,
        xany = i32_xany_fallback_nofma_squared_norm,
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = i32,
        register = Fallback,
        op = generic_sum,
        xconst = i32_xconst_fallback_nofma_sum,
        xany = i32_xany_fallback_nofma_sum,
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = i32,
        register = Fallback,
        op = generic_max_horizontal,
        xconst = i32_xconst_fallback_nofma_max_horizontal,
        xany = i32_xany_fallback_nofma_max_horizontal,
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = i32,
        register = Fallback,
        op = generic_min_horizontal,
        xconst = i32_xconst_fallback_nofma_min_horizontal,
        xany = i32_xany_fallback_nofma_min_horizontal,
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = i32,
        register = Fallback,
        op = generic_max_vertical,
        xconst = i32_xconst_fallback_nofma_max_vertical,
        xany = i32_xany_fallback_nofma_max_vertical,
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = i32,
        register = Fallback,
        op = generic_min_vertical,
        xconst = i32_xconst_fallback_nofma_min_vertical,
        xany = i32_xany_fallback_nofma_min_vertical,
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = i32,
        register = Fallback,
        op = generic_max_value,
        xconst = i32_xconst_fallback_nofma_max_value,
        xany = i32_xany_fallback_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = i32,
        register = Fallback,
        op = generic_min_value,
        xconst = i32_xconst_fallback_nofma_min_value,
        xany = i32_xany_fallback_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = i64,
        register = Fallback,
        op = generic_squared_norm,
        xconst = i64_xconst_fallback_nofma_squared_norm,
        xany = i64_xany_fallback_nofma_squared_norm,
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = i64,
        register = Fallback,
        op = generic_sum,
        xconst = i64_xconst_fallback_nofma_sum,
        xany = i64_xany_fallback_nofma_sum,
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = i64,
        register = Fallback,
        op = generic_max_horizontal,
        xconst = i64_xconst_fallback_nofma_max_horizontal,
        xany = i64_xany_fallback_nofma_max_horizontal,
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = i64,
        register = Fallback,
        op = generic_min_horizontal,
        xconst = i64_xconst_fallback_nofma_min_horizontal,
        xany = i64_xany_fallback_nofma_min_horizontal,
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = i64,
        register = Fallback,
        op = generic_max_vertical,
        xconst = i64_xconst_fallback_nofma_max_vertical,
        xany = i64_xany_fallback_nofma_max_vertical,
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = i64,
        register = Fallback,
        op = generic_min_vertical,
        xconst = i64_xconst_fallback_nofma_min_vertical,
        xany = i64_xany_fallback_nofma_min_vertical,
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = i64,
        register = Fallback,
        op = generic_max_value,
        xconst = i64_xconst_fallback_nofma_max_value,
        xany = i64_xany_fallback_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = i64,
        register = Fallback,
        op = generic_min_value,
        xconst = i64_xconst_fallback_nofma_min_value,
        xany = i64_xany_fallback_nofma_min_value,
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
/// Vector norm ops on the avx2 + fma implementations.
pub mod norm_ops_avx2fma {
    use super::*;

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = f32,
        register = Avx2Fma,
        op = generic_squared_norm,
        xconst = f32_xconst_avx2_fma_squared_norm,
        xany = f32_xany_avx2_fma_squared_norm,
        features = "avx2",
        "fma"
    );
    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = f64,
        register = Avx2Fma,
        op = generic_squared_norm,
        xconst = f64_xconst_avx2_fma_squared_norm,
        xany = f64_xany_avx2_fma_squared_norm,
        features = "avx2",
        "fma"
    );
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
/// Vector min/max/sum/norm ops on the avx2 implementations.
pub mod min_max_sum_norm_ops_avx2 {
    use super::*;

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = f32,
        register = Avx2,
        op = generic_squared_norm,
        xconst = f32_xconst_avx2_nofma_squared_norm,
        xany = f32_xany_avx2_nofma_squared_norm,
        features = "avx2"
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = f32,
        register = Avx2,
        op = generic_sum,
        xconst = f32_xconst_avx2_nofma_sum,
        xany = f32_xany_avx2_nofma_sum,
        features = "avx2"
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = f32,
        register = Avx2,
        op = generic_max_horizontal,
        xconst = f32_xconst_avx2_nofma_max_horizontal,
        xany = f32_xany_avx2_nofma_max_horizontal,
        features = "avx2"
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = f32,
        register = Avx2,
        op = generic_min_horizontal,
        xconst = f32_xconst_avx2_nofma_min_horizontal,
        xany = f32_xany_avx2_nofma_min_horizontal,
        features = "avx2"
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = f32,
        register = Avx2,
        op = generic_max_vertical,
        xconst = f32_xconst_avx2_nofma_max_vertical,
        xany = f32_xany_avx2_nofma_max_vertical,
        features = "avx2"
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = f32,
        register = Avx2,
        op = generic_min_vertical,
        xconst = f32_xconst_avx2_nofma_min_vertical,
        xany = f32_xany_avx2_nofma_min_vertical,
        features = "avx2"
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = f32,
        register = Avx2,
        op = generic_max_value,
        xconst = f32_xconst_avx2_nofma_max_value,
        xany = f32_xany_avx2_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = f32,
        register = Avx2,
        op = generic_min_value,
        xconst = f32_xconst_avx2_nofma_min_value,
        xany = f32_xany_avx2_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = f64,
        register = Avx2,
        op = generic_squared_norm,
        xconst = f64_xconst_avx2_nofma_squared_norm,
        xany = f64_xany_avx2_nofma_squared_norm,
        features = "avx2"
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = f64,
        register = Avx2,
        op = generic_sum,
        xconst = f64_xconst_avx2_nofma_sum,
        xany = f64_xany_avx2_nofma_sum,
        features = "avx2"
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = f64,
        register = Avx2,
        op = generic_max_horizontal,
        xconst = f64_xconst_avx2_nofma_max_horizontal,
        xany = f64_xany_avx2_nofma_max_horizontal,
        features = "avx2"
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = f64,
        register = Avx2,
        op = generic_min_horizontal,
        xconst = f64_xconst_avx2_nofma_min_horizontal,
        xany = f64_xany_avx2_nofma_min_horizontal,
        features = "avx2"
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = f64,
        register = Avx2,
        op = generic_max_vertical,
        xconst = f64_xconst_avx2_nofma_max_vertical,
        xany = f64_xany_avx2_nofma_max_vertical,
        features = "avx2"
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = f64,
        register = Avx2,
        op = generic_min_vertical,
        xconst = f64_xconst_avx2_nofma_min_vertical,
        xany = f64_xany_avx2_nofma_min_vertical,
        features = "avx2"
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = f64,
        register = Avx2,
        op = generic_max_value,
        xconst = f64_xconst_avx2_nofma_max_value,
        xany = f64_xany_avx2_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = f64,
        register = Avx2,
        op = generic_min_value,
        xconst = f64_xconst_avx2_nofma_min_value,
        xany = f64_xany_avx2_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = u8,
        register = Avx2,
        op = generic_squared_norm,
        xconst = u8_xconst_avx2_nofma_squared_norm,
        xany = u8_xany_avx2_nofma_squared_norm,
        features = "avx2"
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = u8,
        register = Avx2,
        op = generic_sum,
        xconst = u8_xconst_avx2_nofma_sum,
        xany = u8_xany_avx2_nofma_sum,
        features = "avx2"
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = u8,
        register = Avx2,
        op = generic_max_horizontal,
        xconst = u8_xconst_avx2_nofma_max_horizontal,
        xany = u8_xany_avx2_nofma_max_horizontal,
        features = "avx2"
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = u8,
        register = Avx2,
        op = generic_min_horizontal,
        xconst = u8_xconst_avx2_nofma_min_horizontal,
        xany = u8_xany_avx2_nofma_min_horizontal,
        features = "avx2"
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = u8,
        register = Avx2,
        op = generic_max_vertical,
        xconst = u8_xconst_avx2_nofma_max_vertical,
        xany = u8_xany_avx2_nofma_max_vertical,
        features = "avx2"
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = u8,
        register = Avx2,
        op = generic_min_vertical,
        xconst = u8_xconst_avx2_nofma_min_vertical,
        xany = u8_xany_avx2_nofma_min_vertical,
        features = "avx2"
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = u8,
        register = Avx2,
        op = generic_max_value,
        xconst = u8_xconst_avx2_nofma_max_value,
        xany = u8_xany_avx2_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = u8,
        register = Avx2,
        op = generic_min_value,
        xconst = u8_xconst_avx2_nofma_min_value,
        xany = u8_xany_avx2_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = u16,
        register = Avx2,
        op = generic_squared_norm,
        xconst = u16_xconst_avx2_nofma_squared_norm,
        xany = u16_xany_avx2_nofma_squared_norm,
        features = "avx2"
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = u16,
        register = Avx2,
        op = generic_sum,
        xconst = u16_xconst_avx2_nofma_sum,
        xany = u16_xany_avx2_nofma_sum,
        features = "avx2"
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = u16,
        register = Avx2,
        op = generic_max_horizontal,
        xconst = u16_xconst_avx2_nofma_max_horizontal,
        xany = u16_xany_avx2_nofma_max_horizontal,
        features = "avx2"
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = u16,
        register = Avx2,
        op = generic_min_horizontal,
        xconst = u16_xconst_avx2_nofma_min_horizontal,
        xany = u16_xany_avx2_nofma_min_horizontal,
        features = "avx2"
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = u16,
        register = Avx2,
        op = generic_max_vertical,
        xconst = u16_xconst_avx2_nofma_max_vertical,
        xany = u16_xany_avx2_nofma_max_vertical,
        features = "avx2"
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = u16,
        register = Avx2,
        op = generic_min_vertical,
        xconst = u16_xconst_avx2_nofma_min_vertical,
        xany = u16_xany_avx2_nofma_min_vertical,
        features = "avx2"
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = u16,
        register = Avx2,
        op = generic_max_value,
        xconst = u16_xconst_avx2_nofma_max_value,
        xany = u16_xany_avx2_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = u16,
        register = Avx2,
        op = generic_min_value,
        xconst = u16_xconst_avx2_nofma_min_value,
        xany = u16_xany_avx2_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = u32,
        register = Avx2,
        op = generic_squared_norm,
        xconst = u32_xconst_avx2_nofma_squared_norm,
        xany = u32_xany_avx2_nofma_squared_norm,
        features = "avx2"
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = u32,
        register = Avx2,
        op = generic_sum,
        xconst = u32_xconst_avx2_nofma_sum,
        xany = u32_xany_avx2_nofma_sum,
        features = "avx2"
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = u32,
        register = Avx2,
        op = generic_max_horizontal,
        xconst = u32_xconst_avx2_nofma_max_horizontal,
        xany = u32_xany_avx2_nofma_max_horizontal,
        features = "avx2"
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = u32,
        register = Avx2,
        op = generic_min_horizontal,
        xconst = u32_xconst_avx2_nofma_min_horizontal,
        xany = u32_xany_avx2_nofma_min_horizontal,
        features = "avx2"
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = u32,
        register = Avx2,
        op = generic_max_vertical,
        xconst = u32_xconst_avx2_nofma_max_vertical,
        xany = u32_xany_avx2_nofma_max_vertical,
        features = "avx2"
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = u32,
        register = Avx2,
        op = generic_min_vertical,
        xconst = u32_xconst_avx2_nofma_min_vertical,
        xany = u32_xany_avx2_nofma_min_vertical,
        features = "avx2"
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = u32,
        register = Avx2,
        op = generic_max_value,
        xconst = u32_xconst_avx2_nofma_max_value,
        xany = u32_xany_avx2_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = u32,
        register = Avx2,
        op = generic_min_value,
        xconst = u32_xconst_avx2_nofma_min_value,
        xany = u32_xany_avx2_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = u64,
        register = Avx2,
        op = generic_squared_norm,
        xconst = u64_xconst_avx2_nofma_squared_norm,
        xany = u64_xany_avx2_nofma_squared_norm,
        features = "avx2"
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = u64,
        register = Avx2,
        op = generic_sum,
        xconst = u64_xconst_avx2_nofma_sum,
        xany = u64_xany_avx2_nofma_sum,
        features = "avx2"
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = u64,
        register = Avx2,
        op = generic_max_horizontal,
        xconst = u64_xconst_avx2_nofma_max_horizontal,
        xany = u64_xany_avx2_nofma_max_horizontal,
        features = "avx2"
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = u64,
        register = Avx2,
        op = generic_min_horizontal,
        xconst = u64_xconst_avx2_nofma_min_horizontal,
        xany = u64_xany_avx2_nofma_min_horizontal,
        features = "avx2"
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = u64,
        register = Avx2,
        op = generic_max_vertical,
        xconst = u64_xconst_avx2_nofma_max_vertical,
        xany = u64_xany_avx2_nofma_max_vertical,
        features = "avx2"
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = u64,
        register = Avx2,
        op = generic_min_vertical,
        xconst = u64_xconst_avx2_nofma_min_vertical,
        xany = u64_xany_avx2_nofma_min_vertical,
        features = "avx2"
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = u64,
        register = Avx2,
        op = generic_max_value,
        xconst = u64_xconst_avx2_nofma_max_value,
        xany = u64_xany_avx2_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = u64,
        register = Avx2,
        op = generic_min_value,
        xconst = u64_xconst_avx2_nofma_min_value,
        xany = u64_xany_avx2_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = i8,
        register = Avx2,
        op = generic_squared_norm,
        xconst = i8_xconst_avx2_nofma_squared_norm,
        xany = i8_xany_avx2_nofma_squared_norm,
        features = "avx2"
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = i8,
        register = Avx2,
        op = generic_sum,
        xconst = i8_xconst_avx2_nofma_sum,
        xany = i8_xany_avx2_nofma_sum,
        features = "avx2"
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = i8,
        register = Avx2,
        op = generic_max_horizontal,
        xconst = i8_xconst_avx2_nofma_max_horizontal,
        xany = i8_xany_avx2_nofma_max_horizontal,
        features = "avx2"
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = i8,
        register = Avx2,
        op = generic_min_horizontal,
        xconst = i8_xconst_avx2_nofma_min_horizontal,
        xany = i8_xany_avx2_nofma_min_horizontal,
        features = "avx2"
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = i8,
        register = Avx2,
        op = generic_max_vertical,
        xconst = i8_xconst_avx2_nofma_max_vertical,
        xany = i8_xany_avx2_nofma_max_vertical,
        features = "avx2"
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = i8,
        register = Avx2,
        op = generic_min_vertical,
        xconst = i8_xconst_avx2_nofma_min_vertical,
        xany = i8_xany_avx2_nofma_min_vertical,
        features = "avx2"
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = i8,
        register = Avx2,
        op = generic_max_value,
        xconst = i8_xconst_avx2_nofma_max_value,
        xany = i8_xany_avx2_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = i8,
        register = Avx2,
        op = generic_min_value,
        xconst = i8_xconst_avx2_nofma_min_value,
        xany = i8_xany_avx2_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = i16,
        register = Avx2,
        op = generic_squared_norm,
        xconst = i16_xconst_avx2_nofma_squared_norm,
        xany = i16_xany_avx2_nofma_squared_norm,
        features = "avx2"
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = i16,
        register = Avx2,
        op = generic_sum,
        xconst = i16_xconst_avx2_nofma_sum,
        xany = i16_xany_avx2_nofma_sum,
        features = "avx2"
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = i16,
        register = Avx2,
        op = generic_max_horizontal,
        xconst = i16_xconst_avx2_nofma_max_horizontal,
        xany = i16_xany_avx2_nofma_max_horizontal,
        features = "avx2"
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = i16,
        register = Avx2,
        op = generic_min_horizontal,
        xconst = i16_xconst_avx2_nofma_min_horizontal,
        xany = i16_xany_avx2_nofma_min_horizontal,
        features = "avx2"
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = i16,
        register = Avx2,
        op = generic_max_vertical,
        xconst = i16_xconst_avx2_nofma_max_vertical,
        xany = i16_xany_avx2_nofma_max_vertical,
        features = "avx2"
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = i16,
        register = Avx2,
        op = generic_min_vertical,
        xconst = i16_xconst_avx2_nofma_min_vertical,
        xany = i16_xany_avx2_nofma_min_vertical,
        features = "avx2"
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = i16,
        register = Avx2,
        op = generic_max_value,
        xconst = i16_xconst_avx2_nofma_max_value,
        xany = i16_xany_avx2_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = i16,
        register = Avx2,
        op = generic_min_value,
        xconst = i16_xconst_avx2_nofma_min_value,
        xany = i16_xany_avx2_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = i32,
        register = Avx2,
        op = generic_squared_norm,
        xconst = i32_xconst_avx2_nofma_squared_norm,
        xany = i32_xany_avx2_nofma_squared_norm,
        features = "avx2"
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = i32,
        register = Avx2,
        op = generic_sum,
        xconst = i32_xconst_avx2_nofma_sum,
        xany = i32_xany_avx2_nofma_sum,
        features = "avx2"
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = i32,
        register = Avx2,
        op = generic_max_horizontal,
        xconst = i32_xconst_avx2_nofma_max_horizontal,
        xany = i32_xany_avx2_nofma_max_horizontal,
        features = "avx2"
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = i32,
        register = Avx2,
        op = generic_min_horizontal,
        xconst = i32_xconst_avx2_nofma_min_horizontal,
        xany = i32_xany_avx2_nofma_min_horizontal,
        features = "avx2"
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = i32,
        register = Avx2,
        op = generic_max_vertical,
        xconst = i32_xconst_avx2_nofma_max_vertical,
        xany = i32_xany_avx2_nofma_max_vertical,
        features = "avx2"
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = i32,
        register = Avx2,
        op = generic_min_vertical,
        xconst = i32_xconst_avx2_nofma_min_vertical,
        xany = i32_xany_avx2_nofma_min_vertical,
        features = "avx2"
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = i32,
        register = Avx2,
        op = generic_max_value,
        xconst = i32_xconst_avx2_nofma_max_value,
        xany = i32_xany_avx2_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = i32,
        register = Avx2,
        op = generic_min_value,
        xconst = i32_xconst_avx2_nofma_min_value,
        xany = i32_xany_avx2_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = i64,
        register = Avx2,
        op = generic_squared_norm,
        xconst = i64_xconst_avx2_nofma_squared_norm,
        xany = i64_xany_avx2_nofma_squared_norm,
        features = "avx2"
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = i64,
        register = Avx2,
        op = generic_sum,
        xconst = i64_xconst_avx2_nofma_sum,
        xany = i64_xany_avx2_nofma_sum,
        features = "avx2"
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = i64,
        register = Avx2,
        op = generic_max_horizontal,
        xconst = i64_xconst_avx2_nofma_max_horizontal,
        xany = i64_xany_avx2_nofma_max_horizontal,
        features = "avx2"
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = i64,
        register = Avx2,
        op = generic_min_horizontal,
        xconst = i64_xconst_avx2_nofma_min_horizontal,
        xany = i64_xany_avx2_nofma_min_horizontal,
        features = "avx2"
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = i64,
        register = Avx2,
        op = generic_max_vertical,
        xconst = i64_xconst_avx2_nofma_max_vertical,
        xany = i64_xany_avx2_nofma_max_vertical,
        features = "avx2"
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = i64,
        register = Avx2,
        op = generic_min_vertical,
        xconst = i64_xconst_avx2_nofma_min_vertical,
        xany = i64_xany_avx2_nofma_min_vertical,
        features = "avx2"
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = i64,
        register = Avx2,
        op = generic_max_value,
        xconst = i64_xconst_avx2_nofma_max_value,
        xany = i64_xany_avx2_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = i64,
        register = Avx2,
        op = generic_min_value,
        xconst = i64_xconst_avx2_nofma_min_value,
        xany = i64_xany_avx2_nofma_min_value,
    );
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
/// Vector min/max/sum ops on the avx512 implementations.
///
/// These methods do not strictly require any CPU feature but do auto-vectorize
/// well when target features are explicitly provided and can be used to
/// provide accelerated operations on non-explicitly supported SIMD architectures.
pub mod min_max_sum_norm_ops_avx512 {
    use super::*;

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = f32,
        register = Avx512,
        op = generic_squared_norm,
        xconst = f32_xconst_avx512_fma_squared_norm,
        xany = f32_xany_avx512_fma_squared_norm,
        features = "avx512f"
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = f32,
        register = Avx512,
        op = generic_sum,
        xconst = f32_xconst_avx512_nofma_sum,
        xany = f32_xany_avx512_nofma_sum,
        features = "avx512f"
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = f32,
        register = Avx512,
        op = generic_max_horizontal,
        xconst = f32_xconst_avx512_nofma_max_horizontal,
        xany = f32_xany_avx512_nofma_max_horizontal,
        features = "avx512f"
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = f32,
        register = Avx512,
        op = generic_min_horizontal,
        xconst = f32_xconst_avx512_nofma_min_horizontal,
        xany = f32_xany_avx512_nofma_min_horizontal,
        features = "avx512f"
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = f32,
        register = Avx512,
        op = generic_max_vertical,
        xconst = f32_xconst_avx512_nofma_max_vertical,
        xany = f32_xany_avx512_nofma_max_vertical,
        features = "avx512f"
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = f32,
        register = Avx512,
        op = generic_min_vertical,
        xconst = f32_xconst_avx512_nofma_min_vertical,
        xany = f32_xany_avx512_nofma_min_vertical,
        features = "avx512f"
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = f32,
        register = Avx512,
        op = generic_max_value,
        xconst = f32_xconst_avx512_nofma_max_value,
        xany = f32_xany_avx512_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = f32,
        register = Avx512,
        op = generic_min_value,
        xconst = f32_xconst_avx512_nofma_min_value,
        xany = f32_xany_avx512_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = f64,
        register = Avx512,
        op = generic_squared_norm,
        xconst = f64_xconst_avx512_fma_squared_norm,
        xany = f64_xany_avx512_fma_squared_norm,
        features = "avx512f"
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = f64,
        register = Avx512,
        op = generic_sum,
        xconst = f64_xconst_avx512_nofma_sum,
        xany = f64_xany_avx512_nofma_sum,
        features = "avx512f"
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = f64,
        register = Avx512,
        op = generic_max_horizontal,
        xconst = f64_xconst_avx512_nofma_max_horizontal,
        xany = f64_xany_avx512_nofma_max_horizontal,
        features = "avx512f"
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = f64,
        register = Avx512,
        op = generic_min_horizontal,
        xconst = f64_xconst_avx512_nofma_min_horizontal,
        xany = f64_xany_avx512_nofma_min_horizontal,
        features = "avx512f"
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = f64,
        register = Avx512,
        op = generic_max_vertical,
        xconst = f64_xconst_avx512_nofma_max_vertical,
        xany = f64_xany_avx512_nofma_max_vertical,
        features = "avx512f"
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = f64,
        register = Avx512,
        op = generic_max_vertical,
        xconst = f64_xconst_avx512_nofma_min_vertical,
        xany = f64_xany_avx512_nofma_min_vertical,
        features = "avx512f"
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = f64,
        register = Avx512,
        op = generic_max_value,
        xconst = f64_xconst_avx512_nofma_max_value,
        xany = f64_xany_avx512_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = f64,
        register = Avx512,
        op = generic_min_value,
        xconst = f64_xconst_avx512_nofma_min_value,
        xany = f64_xany_avx512_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = u8,
        register = Avx512,
        op = generic_squared_norm,
        xconst = u8_xconst_avx512_nofma_squared_norm,
        xany = u8_xany_avx512_nofma_squared_norm,
        features = "avx512f"
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = u8,
        register = Avx512,
        op = generic_sum,
        xconst = u8_xconst_avx512_nofma_sum,
        xany = u8_xany_avx512_nofma_sum,
        features = "avx512f"
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = u8,
        register = Avx512,
        op = generic_max_horizontal,
        xconst = u8_xconst_avx512_nofma_max_horizontal,
        xany = u8_xany_avx512_nofma_max_horizontal,
        features = "avx512f"
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = u8,
        register = Avx512,
        op = generic_min_horizontal,
        xconst = u8_xconst_avx512_nofma_min_horizontal,
        xany = u8_xany_avx512_nofma_min_horizontal,
        features = "avx512f"
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = u8,
        register = Avx512,
        op = generic_max_vertical,
        xconst = u8_xconst_avx512_nofma_max_vertical,
        xany = u8_xany_avx512_nofma_max_vertical,
        features = "avx512f"
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = u8,
        register = Avx512,
        op = generic_min_vertical,
        xconst = u8_xconst_avx512_nofma_min_vertical,
        xany = u8_xany_avx512_nofma_min_vertical,
        features = "avx512f"
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = u8,
        register = Avx512,
        op = generic_max_value,
        xconst = u8_xconst_avx512_nofma_max_value,
        xany = u8_xany_avx512_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = u8,
        register = Avx512,
        op = generic_min_value,
        xconst = u8_xconst_avx512_nofma_min_value,
        xany = u8_xany_avx512_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = u16,
        register = Avx512,
        op = generic_squared_norm,
        xconst = u16_xconst_avx512_nofma_squared_norm,
        xany = u16_xany_avx512_nofma_squared_norm,
        features = "avx512f"
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = u16,
        register = Avx512,
        op = generic_sum,
        xconst = u16_xconst_avx512_nofma_sum,
        xany = u16_xany_avx512_nofma_sum,
        features = "avx512f"
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = u16,
        register = Avx512,
        op = generic_max_horizontal,
        xconst = u16_xconst_avx512_nofma_max_horizontal,
        xany = u16_xany_avx512_nofma_max_horizontal,
        features = "avx512f"
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = u16,
        register = Avx512,
        op = generic_min_horizontal,
        xconst = u16_xconst_avx512_nofma_min_horizontal,
        xany = u16_xany_avx512_nofma_min_horizontal,
        features = "avx512f"
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = u16,
        register = Avx512,
        op = generic_max_vertical,
        xconst = u16_xconst_avx512_nofma_max_vertical,
        xany = u16_xany_avx512_nofma_max_vertical,
        features = "avx512f"
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = u16,
        register = Avx512,
        op = generic_min_vertical,
        xconst = u16_xconst_avx512_nofma_min_vertical,
        xany = u16_xany_avx512_nofma_min_vertical,
        features = "avx512f"
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = u16,
        register = Avx512,
        op = generic_max_value,
        xconst = u16_xconst_avx512_nofma_max_value,
        xany = u16_xany_avx512_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = u16,
        register = Avx512,
        op = generic_min_value,
        xconst = u16_xconst_avx512_nofma_min_value,
        xany = u16_xany_avx512_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = u32,
        register = Avx512,
        op = generic_squared_norm,
        xconst = u32_xconst_avx512_nofma_squared_norm,
        xany = u32_xany_avx512_nofma_squared_norm,
        features = "avx512f"
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = u32,
        register = Avx512,
        op = generic_sum,
        xconst = u32_xconst_avx512_nofma_sum,
        xany = u32_xany_avx512_nofma_sum,
        features = "avx512f"
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = u32,
        register = Avx512,
        op = generic_max_horizontal,
        xconst = u32_xconst_avx512_nofma_max_horizontal,
        xany = u32_xany_avx512_nofma_max_horizontal,
        features = "avx512f"
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = u32,
        register = Avx512,
        op = generic_min_horizontal,
        xconst = u32_xconst_avx512_nofma_min_horizontal,
        xany = u32_xany_avx512_nofma_min_horizontal,
        features = "avx512f"
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = u32,
        register = Avx512,
        op = generic_max_vertical,
        xconst = u32_xconst_avx512_nofma_max_vertical,
        xany = u32_xany_avx512_nofma_max_vertical,
        features = "avx512f"
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = u32,
        register = Avx512,
        op = generic_min_vertical,
        xconst = u32_xconst_avx512_nofma_min_vertical,
        xany = u32_xany_avx512_nofma_min_vertical,
        features = "avx512f"
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = u32,
        register = Avx512,
        op = generic_max_value,
        xconst = u32_xconst_avx512_nofma_max_value,
        xany = u32_xany_avx512_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = u32,
        register = Avx512,
        op = generic_min_value,
        xconst = u32_xconst_avx512_nofma_min_value,
        xany = u32_xany_avx512_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = u64,
        register = Avx512,
        op = generic_squared_norm,
        xconst = u64_xconst_avx512_nofma_squared_norm,
        xany = u64_xany_avx512_nofma_squared_norm,
        features = "avx512f"
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = u64,
        register = Avx512,
        op = generic_sum,
        xconst = u64_xconst_avx512_nofma_sum,
        xany = u64_xany_avx512_nofma_sum,
        features = "avx512f"
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = u64,
        register = Avx512,
        op = generic_max_horizontal,
        xconst = u64_xconst_avx512_nofma_max_horizontal,
        xany = u64_xany_avx512_nofma_max_horizontal,
        features = "avx512f"
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = u64,
        register = Avx512,
        op = generic_min_horizontal,
        xconst = u64_xconst_avx512_nofma_min_horizontal,
        xany = u64_xany_avx512_nofma_min_horizontal,
        features = "avx512f"
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = u64,
        register = Avx512,
        op = generic_max_vertical,
        xconst = u64_xconst_avx512_nofma_max_vertical,
        xany = u64_xany_avx512_nofma_max_vertical,
        features = "avx512f"
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = u64,
        register = Avx512,
        op = generic_min_vertical,
        xconst = u64_xconst_avx512_nofma_min_vertical,
        xany = u64_xany_avx512_nofma_min_vertical,
        features = "avx512f"
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = u64,
        register = Avx512,
        op = generic_max_value,
        xconst = u64_xconst_avx512_nofma_max_value,
        xany = u64_xany_avx512_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = u64,
        register = Avx512,
        op = generic_min_value,
        xconst = u64_xconst_avx512_nofma_min_value,
        xany = u64_xany_avx512_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = i8,
        register = Avx512,
        op = generic_squared_norm,
        xconst = i8_xconst_avx512_nofma_squared_norm,
        xany = i8_xany_avx512_nofma_squared_norm,
        features = "avx512f"
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = i8,
        register = Avx512,
        op = generic_sum,
        xconst = i8_xconst_avx512_nofma_sum,
        xany = i8_xany_avx512_nofma_sum,
        features = "avx512f"
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = i8,
        register = Avx512,
        op = generic_max_horizontal,
        xconst = i8_xconst_avx512_nofma_max_horizontal,
        xany = i8_xany_avx512_nofma_max_horizontal,
        features = "avx512f"
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = i8,
        register = Avx512,
        op = generic_min_horizontal,
        xconst = i8_xconst_avx512_nofma_min_horizontal,
        xany = i8_xany_avx512_nofma_min_horizontal,
        features = "avx512f"
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = i8,
        register = Avx512,
        op = generic_max_vertical,
        xconst = i8_xconst_avx512_nofma_max_vertical,
        xany = i8_xany_avx512_nofma_max_vertical,
        features = "avx512f"
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = i8,
        register = Avx512,
        op = generic_min_vertical,
        xconst = i8_xconst_avx512_nofma_min_vertical,
        xany = i8_xany_avx512_nofma_min_vertical,
        features = "avx512f"
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = i8,
        register = Avx512,
        op = generic_max_value,
        xconst = i8_xconst_avx512_nofma_max_value,
        xany = i8_xany_avx512_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = i8,
        register = Avx512,
        op = generic_min_value,
        xconst = i8_xconst_avx512_nofma_min_value,
        xany = i8_xany_avx512_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = i16,
        register = Avx512,
        op = generic_squared_norm,
        xconst = i16_xconst_avx512_nofma_squared_norm,
        xany = i16_xany_avx512_nofma_squared_norm,
        features = "avx512f"
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = i16,
        register = Avx512,
        op = generic_sum,
        xconst = i16_xconst_avx512_nofma_sum,
        xany = i16_xany_avx512_nofma_sum,
        features = "avx512f"
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = i16,
        register = Avx512,
        op = generic_max_horizontal,
        xconst = i16_xconst_avx512_nofma_max_horizontal,
        xany = i16_xany_avx512_nofma_max_horizontal,
        features = "avx512f"
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = i16,
        register = Avx512,
        op = generic_min_horizontal,
        xconst = i16_xconst_avx512_nofma_min_horizontal,
        xany = i16_xany_avx512_nofma_min_horizontal,
        features = "avx512f"
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = i16,
        register = Avx512,
        op = generic_max_vertical,
        xconst = i16_xconst_avx512_nofma_max_vertical,
        xany = i16_xany_avx512_nofma_max_vertical,
        features = "avx512f"
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = i16,
        register = Avx512,
        op = generic_min_vertical,
        xconst = i16_xconst_avx512_nofma_min_vertical,
        xany = i16_xany_avx512_nofma_min_vertical,
        features = "avx512f"
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = i16,
        register = Avx512,
        op = generic_max_value,
        xconst = i16_xconst_avx512_nofma_max_value,
        xany = i16_xany_avx512_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = i16,
        register = Avx512,
        op = generic_min_value,
        xconst = i16_xconst_avx512_nofma_min_value,
        xany = i16_xany_avx512_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = i32,
        register = Avx512,
        op = generic_squared_norm,
        xconst = i32_xconst_avx512_nofma_squared_norm,
        xany = i32_xany_avx512_nofma_squared_norm,
        features = "avx512f"
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = i32,
        register = Avx512,
        op = generic_sum,
        xconst = i32_xconst_avx512_nofma_sum,
        xany = i32_xany_avx512_nofma_sum,
        features = "avx512f"
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = i32,
        register = Avx512,
        op = generic_max_horizontal,
        xconst = i32_xconst_avx512_nofma_max_horizontal,
        xany = i32_xany_avx512_nofma_max_horizontal,
        features = "avx512f"
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = i32,
        register = Avx512,
        op = generic_min_horizontal,
        xconst = i32_xconst_avx512_nofma_min_horizontal,
        xany = i32_xany_avx512_nofma_min_horizontal,
        features = "avx512f"
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = i32,
        register = Avx512,
        op = generic_max_vertical,
        xconst = i32_xconst_avx512_nofma_max_vertical,
        xany = i32_xany_avx512_nofma_max_vertical,
        features = "avx512f"
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = i32,
        register = Avx512,
        op = generic_min_vertical,
        xconst = i32_xconst_avx512_nofma_min_vertical,
        xany = i32_xany_avx512_nofma_min_vertical,
        features = "avx512f"
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = i32,
        register = Avx512,
        op = generic_max_value,
        xconst = i32_xconst_avx512_nofma_max_value,
        xany = i32_xany_avx512_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = i32,
        register = Avx512,
        op = generic_min_value,
        xconst = i32_xconst_avx512_nofma_min_value,
        xany = i32_xany_avx512_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = i64,
        register = Avx512,
        op = generic_squared_norm,
        xconst = i64_xconst_avx512_nofma_squared_norm,
        xany = i64_xany_avx512_nofma_squared_norm,
        features = "avx512f"
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = i64,
        register = Avx512,
        op = generic_sum,
        xconst = i64_xconst_avx512_nofma_sum,
        xany = i64_xany_avx512_nofma_sum,
        features = "avx512f"
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = i64,
        register = Avx512,
        op = generic_max_horizontal,
        xconst = i64_xconst_avx512_nofma_max_horizontal,
        xany = i64_xany_avx512_nofma_max_horizontal,
        features = "avx512f"
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = i64,
        register = Avx512,
        op = generic_min_horizontal,
        xconst = i64_xconst_avx512_nofma_min_horizontal,
        xany = i64_xany_avx512_nofma_min_horizontal,
        features = "avx512f"
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = i64,
        register = Avx512,
        op = generic_max_vertical,
        xconst = i64_xconst_avx512_nofma_max_vertical,
        xany = i64_xany_avx512_nofma_max_vertical,
        features = "avx512f"
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = i64,
        register = Avx512,
        op = generic_min_vertical,
        xconst = i64_xconst_avx512_nofma_min_vertical,
        xany = i64_xany_avx512_nofma_min_vertical,
        features = "avx512f"
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = i64,
        register = Avx512,
        op = generic_max_value,
        xconst = i64_xconst_avx512_nofma_max_value,
        xany = i64_xany_avx512_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = i64,
        register = Avx512,
        op = generic_min_value,
        xconst = i64_xconst_avx512_nofma_min_value,
        xany = i64_xany_avx512_nofma_min_value,
    );
}

#[cfg(target_arch = "aarch64")]
/// Vector min/max/sum/norm ops on the neon implementations.
pub mod min_max_sum_norm_ops_neon {
    use super::*;

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = f32,
        register = Neon,
        op = generic_squared_norm,
        xconst = f32_xconst_neon_fma_squared_norm,
        xany = f32_xany_neon_fma_squared_norm,
        features = "neon"
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = f32,
        register = Neon,
        op = generic_sum,
        xconst = f32_xconst_neon_nofma_sum,
        xany = f32_xany_neon_nofma_sum,
        features = "neon"
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = f32,
        register = Neon,
        op = generic_max_horizontal,
        xconst = f32_xconst_neon_nofma_max_horizontal,
        xany = f32_xany_neon_nofma_max_horizontal,
        features = "neon"
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = f32,
        register = Neon,
        op = generic_min_horizontal,
        xconst = f32_xconst_neon_nofma_min_horizontal,
        xany = f32_xany_neon_nofma_min_horizontal,
        features = "neon"
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = f32,
        register = Neon,
        op = generic_max_vertical,
        xconst = f32_xconst_neon_nofma_max_vertical,
        xany = f32_xany_neon_nofma_max_vertical,
        features = "neon"
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = f32,
        register = Neon,
        op = generic_min_vertical,
        xconst = f32_xconst_neon_nofma_min_vertical,
        xany = f32_xany_neon_nofma_min_vertical,
        features = "neon"
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = f32,
        register = Neon,
        op = generic_max_value,
        xconst = f32_xconst_neon_nofma_max_value,
        xany = f32_xany_neon_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = f32,
        register = Neon,
        op = generic_min_value,
        xconst = f32_xconst_neon_nofma_min_value,
        xany = f32_xany_neon_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = f64,
        register = Neon,
        op = generic_squared_norm,
        xconst = f64_xconst_neon_fma_squared_norm,
        xany = f64_xany_neon_fma_squared_norm,
        features = "neon"
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = f64,
        register = Neon,
        op = generic_sum,
        xconst = f64_xconst_neon_nofma_sum,
        xany = f64_xany_neon_nofma_sum,
        features = "neon"
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = f64,
        register = Neon,
        op = generic_max_horizontal,
        xconst = f64_xconst_neon_nofma_max_horizontal,
        xany = f64_xany_neon_nofma_max_horizontal,
        features = "neon"
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = f64,
        register = Neon,
        op = generic_min_horizontal,
        xconst = f64_xconst_neon_nofma_min_horizontal,
        xany = f64_xany_neon_nofma_min_horizontal,
        features = "neon"
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = f64,
        register = Neon,
        op = generic_max_vertical,
        xconst = f64_xconst_neon_nofma_max_vertical,
        xany = f64_xany_neon_nofma_max_vertical,
        features = "neon"
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = f64,
        register = Neon,
        op = generic_min_vertical,
        xconst = f64_xconst_neon_nofma_min_vertical,
        xany = f64_xany_neon_nofma_min_vertical,
        features = "neon"
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = f64,
        register = Neon,
        op = generic_max_value,
        xconst = f64_xconst_neon_nofma_max_value,
        xany = f64_xany_neon_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = f64,
        register = Neon,
        op = generic_min_value,
        xconst = f64_xconst_neon_nofma_min_value,
        xany = f64_xany_neon_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = u8,
        register = Neon,
        op = generic_squared_norm,
        xconst = u8_xconst_neon_nofma_squared_norm,
        xany = u8_xany_neon_nofma_squared_norm,
        features = "neon"
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = u8,
        register = Neon,
        op = generic_sum,
        xconst = u8_xconst_neon_nofma_sum,
        xany = u8_xany_neon_nofma_sum,
        features = "neon"
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = u8,
        register = Neon,
        op = generic_max_horizontal,
        xconst = u8_xconst_neon_nofma_max_horizontal,
        xany = u8_xany_neon_nofma_max_horizontal,
        features = "neon"
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = u8,
        register = Neon,
        op = generic_min_horizontal,
        xconst = u8_xconst_neon_nofma_min_horizontal,
        xany = u8_xany_neon_nofma_min_horizontal,
        features = "neon"
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = u8,
        register = Neon,
        op = generic_max_vertical,
        xconst = u8_xconst_neon_nofma_max_vertical,
        xany = u8_xany_neon_nofma_max_vertical,
        features = "neon"
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = u8,
        register = Neon,
        op = generic_min_vertical,
        xconst = u8_xconst_neon_nofma_min_vertical,
        xany = u8_xany_neon_nofma_min_vertical,
        features = "neon"
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = u8,
        register = Neon,
        op = generic_max_value,
        xconst = u8_xconst_neon_nofma_max_value,
        xany = u8_xany_neon_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = u8,
        register = Neon,
        op = generic_min_value,
        xconst = u8_xconst_neon_nofma_min_value,
        xany = u8_xany_neon_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = u16,
        register = Neon,
        op = generic_squared_norm,
        xconst = u16_xconst_neon_nofma_squared_norm,
        xany = u16_xany_neon_nofma_squared_norm,
        features = "neon"
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = u16,
        register = Neon,
        op = generic_sum,
        xconst = u16_xconst_neon_nofma_sum,
        xany = u16_xany_neon_nofma_sum,
        features = "neon"
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = u16,
        register = Neon,
        op = generic_max_horizontal,
        xconst = u16_xconst_neon_nofma_max_horizontal,
        xany = u16_xany_neon_nofma_max_horizontal,
        features = "neon"
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = u16,
        register = Neon,
        op = generic_min_horizontal,
        xconst = u16_xconst_neon_nofma_min_horizontal,
        xany = u16_xany_neon_nofma_min_horizontal,
        features = "neon"
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = u16,
        register = Neon,
        op = generic_max_vertical,
        xconst = u16_xconst_neon_nofma_max_vertical,
        xany = u16_xany_neon_nofma_max_vertical,
        features = "neon"
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = u16,
        register = Neon,
        op = generic_min_vertical,
        xconst = u16_xconst_neon_nofma_min_vertical,
        xany = u16_xany_neon_nofma_min_vertical,
        features = "neon"
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = u16,
        register = Neon,
        op = generic_max_value,
        xconst = u16_xconst_neon_nofma_max_value,
        xany = u16_xany_neon_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = u16,
        register = Neon,
        op = generic_min_value,
        xconst = u16_xconst_neon_nofma_min_value,
        xany = u16_xany_neon_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = u32,
        register = Neon,
        op = generic_squared_norm,
        xconst = u32_xconst_neon_nofma_squared_norm,
        xany = u32_xany_neon_nofma_squared_norm,
        features = "neon"
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = u32,
        register = Neon,
        op = generic_sum,
        xconst = u32_xconst_neon_nofma_sum,
        xany = u32_xany_neon_nofma_sum,
        features = "neon"
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = u32,
        register = Neon,
        op = generic_max_horizontal,
        xconst = u32_xconst_neon_nofma_max_horizontal,
        xany = u32_xany_neon_nofma_max_horizontal,
        features = "neon"
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = u32,
        register = Neon,
        op = generic_min_horizontal,
        xconst = u32_xconst_neon_nofma_min_horizontal,
        xany = u32_xany_neon_nofma_min_horizontal,
        features = "neon"
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = u32,
        register = Neon,
        op = generic_max_vertical,
        xconst = u32_xconst_neon_nofma_max_vertical,
        xany = u32_xany_neon_nofma_max_vertical,
        features = "neon"
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = u32,
        register = Neon,
        op = generic_min_vertical,
        xconst = u32_xconst_neon_nofma_min_vertical,
        xany = u32_xany_neon_nofma_min_vertical,
        features = "neon"
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = u32,
        register = Neon,
        op = generic_max_value,
        xconst = u32_xconst_neon_nofma_max_value,
        xany = u32_xany_neon_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = u32,
        register = Neon,
        op = generic_min_value,
        xconst = u32_xconst_neon_nofma_min_value,
        xany = u32_xany_neon_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = u64,
        register = Neon,
        op = generic_squared_norm,
        xconst = u64_xconst_neon_nofma_squared_norm,
        xany = u64_xany_neon_nofma_squared_norm,
        features = "neon"
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = u64,
        register = Neon,
        op = generic_sum,
        xconst = u64_xconst_neon_nofma_sum,
        xany = u64_xany_neon_nofma_sum,
        features = "neon"
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = u64,
        register = Neon,
        op = generic_max_horizontal,
        xconst = u64_xconst_neon_nofma_max_horizontal,
        xany = u64_xany_neon_nofma_max_horizontal,
        features = "neon"
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = u64,
        register = Neon,
        op = generic_min_horizontal,
        xconst = u64_xconst_neon_nofma_min_horizontal,
        xany = u64_xany_neon_nofma_min_horizontal,
        features = "neon"
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = u64,
        register = Neon,
        op = generic_max_vertical,
        xconst = u64_xconst_neon_nofma_max_vertical,
        xany = u64_xany_neon_nofma_max_vertical,
        features = "neon"
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = u64,
        register = Neon,
        op = generic_min_vertical,
        xconst = u64_xconst_neon_nofma_min_vertical,
        xany = u64_xany_neon_nofma_min_vertical,
        features = "neon"
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = u64,
        register = Neon,
        op = generic_max_value,
        xconst = u64_xconst_neon_nofma_max_value,
        xany = u64_xany_neon_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = u64,
        register = Neon,
        op = generic_min_value,
        xconst = u64_xconst_neon_nofma_min_value,
        xany = u64_xany_neon_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = i8,
        register = Neon,
        op = generic_squared_norm,
        xconst = i8_xconst_neon_nofma_squared_norm,
        xany = i8_xany_neon_nofma_squared_norm,
        features = "neon"
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = i8,
        register = Neon,
        op = generic_sum,
        xconst = i8_xconst_neon_nofma_sum,
        xany = i8_xany_neon_nofma_sum,
        features = "neon"
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = i8,
        register = Neon,
        op = generic_max_horizontal,
        xconst = i8_xconst_neon_nofma_max_horizontal,
        xany = i8_xany_neon_nofma_max_horizontal,
        features = "neon"
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = i8,
        register = Neon,
        op = generic_min_horizontal,
        xconst = i8_xconst_neon_nofma_min_horizontal,
        xany = i8_xany_neon_nofma_min_horizontal,
        features = "neon"
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = i8,
        register = Neon,
        op = generic_max_vertical,
        xconst = i8_xconst_neon_nofma_max_vertical,
        xany = i8_xany_neon_nofma_max_vertical,
        features = "neon"
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = i8,
        register = Neon,
        op = generic_min_vertical,
        xconst = i8_xconst_neon_nofma_min_vertical,
        xany = i8_xany_neon_nofma_min_vertical,
        features = "neon"
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = i8,
        register = Neon,
        op = generic_max_value,
        xconst = i8_xconst_neon_nofma_max_value,
        xany = i8_xany_neon_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = i8,
        register = Neon,
        op = generic_min_value,
        xconst = i8_xconst_neon_nofma_min_value,
        xany = i8_xany_neon_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = i16,
        register = Neon,
        op = generic_squared_norm,
        xconst = i16_xconst_neon_nofma_squared_norm,
        xany = i16_xany_neon_nofma_squared_norm,
        features = "neon"
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = i16,
        register = Neon,
        op = generic_sum,
        xconst = i16_xconst_neon_nofma_sum,
        xany = i16_xany_neon_nofma_sum,
        features = "neon"
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = i16,
        register = Neon,
        op = generic_max_horizontal,
        xconst = i16_xconst_neon_nofma_max_horizontal,
        xany = i16_xany_neon_nofma_max_horizontal,
        features = "neon"
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = i16,
        register = Neon,
        op = generic_min_horizontal,
        xconst = i16_xconst_neon_nofma_min_horizontal,
        xany = i16_xany_neon_nofma_min_horizontal,
        features = "neon"
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = i16,
        register = Neon,
        op = generic_max_vertical,
        xconst = i16_xconst_neon_nofma_max_vertical,
        xany = i16_xany_neon_nofma_max_vertical,
        features = "neon"
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = i16,
        register = Neon,
        op = generic_min_vertical,
        xconst = i16_xconst_neon_nofma_min_vertical,
        xany = i16_xany_neon_nofma_min_vertical,
        features = "neon"
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = i16,
        register = Neon,
        op = generic_max_value,
        xconst = i16_xconst_neon_nofma_max_value,
        xany = i16_xany_neon_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = i16,
        register = Neon,
        op = generic_min_value,
        xconst = i16_xconst_neon_nofma_min_value,
        xany = i16_xany_neon_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = i32,
        register = Neon,
        op = generic_squared_norm,
        xconst = i32_xconst_neon_nofma_squared_norm,
        xany = i32_xany_neon_nofma_squared_norm,
        features = "neon"
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = i32,
        register = Neon,
        op = generic_sum,
        xconst = i32_xconst_neon_nofma_sum,
        xany = i32_xany_neon_nofma_sum,
        features = "neon"
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = i32,
        register = Neon,
        op = generic_max_horizontal,
        xconst = i32_xconst_neon_nofma_max_horizontal,
        xany = i32_xany_neon_nofma_max_horizontal,
        features = "neon"
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = i32,
        register = Neon,
        op = generic_min_horizontal,
        xconst = i32_xconst_neon_nofma_min_horizontal,
        xany = i32_xany_neon_nofma_min_horizontal,
        features = "neon"
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = i32,
        register = Neon,
        op = generic_max_vertical,
        xconst = i32_xconst_neon_nofma_max_vertical,
        xany = i32_xany_neon_nofma_max_vertical,
        features = "neon"
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = i32,
        register = Neon,
        op = generic_min_vertical,
        xconst = i32_xconst_neon_nofma_min_vertical,
        xany = i32_xany_neon_nofma_min_vertical,
        features = "neon"
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = i32,
        register = Neon,
        op = generic_max_value,
        xconst = i32_xconst_neon_nofma_max_value,
        xany = i32_xany_neon_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = i32,
        register = Neon,
        op = generic_min_value,
        xconst = i32_xconst_neon_nofma_min_value,
        xany = i32_xany_neon_nofma_min_value,
    );

    export_op_horizontal!(
        description = "Calculates the squared L2 norm of the provided vector",
        ty = i64,
        register = Neon,
        op = generic_squared_norm,
        xconst = i64_xconst_neon_nofma_squared_norm,
        xany = i64_xany_neon_nofma_squared_norm,
        features = "neon"
    );
    export_op_horizontal!(
        description = "Horizontal sum of the provided vector",
        ty = i64,
        register = Neon,
        op = generic_sum,
        xconst = i64_xconst_neon_nofma_sum,
        xany = i64_xany_neon_nofma_sum,
        features = "neon"
    );
    export_op_horizontal!(
        description = "Horizontal max of the provided vector",
        ty = i64,
        register = Neon,
        op = generic_max_horizontal,
        xconst = i64_xconst_neon_nofma_max_horizontal,
        xany = i64_xany_neon_nofma_max_horizontal,
        features = "neon"
    );
    export_op_horizontal!(
        description = "Horizontal min of the provided vector",
        ty = i64,
        register = Neon,
        op = generic_min_horizontal,
        xconst = i64_xconst_neon_nofma_min_horizontal,
        xany = i64_xany_neon_nofma_min_horizontal,
        features = "neon"
    );
    export_op_vertical!(
        description = "Vertical max of the two provided vectors",
        ty = i64,
        register = Neon,
        op = generic_max_vertical,
        xconst = i64_xconst_neon_nofma_max_vertical,
        xany = i64_xany_neon_nofma_max_vertical,
        features = "neon"
    );
    export_op_vertical!(
        description = "Vertical min of the two provided vectors",
        ty = i64,
        register = Neon,
        op = generic_min_vertical,
        xconst = i64_xconst_neon_nofma_min_vertical,
        xany = i64_xany_neon_nofma_min_vertical,
        features = "neon"
    );
    export_op_value!(
        description = "Vertical max of a provided vector and a single broadcast value",
        ty = i64,
        register = Neon,
        op = generic_max_value,
        xconst = i64_xconst_neon_nofma_max_value,
        xany = i64_xany_neon_nofma_max_value,
    );
    export_op_value!(
        description = "Vertical min of a provided vector and a single broadcast value",
        ty = i64,
        register = Neon,
        op = generic_min_value,
        xconst = i64_xconst_neon_nofma_min_value,
        xany = i64_xany_neon_nofma_min_value,
    );
}
