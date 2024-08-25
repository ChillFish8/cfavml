//! Safe but somewhat low-level variants of the comparison operations in CFAVML.
//!
//! In general, I would recommend using the higher level generic functions api which provides
//! some syntax sugar over these traits.

use crate::buffer::WriteOnlyBuffer;
use crate::danger::export_cmp_ops;
use crate::mem_loader::{IntoMemLoader, MemLoader};

/// Various comparison operations over vectors.
pub trait CmpOps: Sized {
    /// Finds the horizontal max element of a given vector and returns the result.
    ///
    /// ### Implementation Pseudocode
    ///
    /// ```ignore
    /// result = -inf
    ///
    /// for i in range(dims):
    ///     result = max(result, a[i])
    ///
    /// return result
    /// ```
    ///
    /// ### Panics
    ///
    /// Panics if the size of vector `a` does not match `dims`.
    fn max<B1>(a: B1) -> Self
    where
        B1: IntoMemLoader<Self>,
        B1::Loader: MemLoader<Value = Self>;

    /// Performs an element wise max on each element of vector `a` and `b`,
    /// writing the result to `result`.
    ///
    /// ### Implementation Pseudocode
    ///
    /// ```ignore
    /// result = [0; dims]
    ///
    /// for i in range(dims):
    ///     result[i] = max(a[i], b[i])
    ///
    /// return result
    /// ```
    ///
    /// ### Result buffer
    ///
    /// The result buffer can be either an initialized slice i.e. `&mut [Self]`
    /// or it can be a slice holding potentially uninitialized data i.e. `&mut [MaybeUninit<Self>]`.
    ///
    /// Once the operation is complete, it is safe to assume the data written is fully initialized.
    ///
    /// ### Panics
    ///
    /// Panics if the size of vector `a`, `b` or `result` does not match `dims`.
    fn max_vertical<B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
    where
        B1: IntoMemLoader<Self>,
        B1::Loader: MemLoader<Value = Self>,
        B2: IntoMemLoader<Self>,
        B2::Loader: MemLoader<Value = Self>,
        for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = Self>;

    /// Finds the horizontal min element of a given vector.
    ///
    /// ### Implementation Pseudocode
    ///
    /// ```ignore
    /// result = inf
    ///
    /// for i in range(dims):
    ///     result = min(result, a[i])
    ///
    /// return result
    /// ```
    fn min<B1>(a: B1) -> Self
    where
        B1: IntoMemLoader<Self>,
        B1::Loader: MemLoader<Value = Self>;

    /// Performs an element wise min on each element of vector `a` and `b`,
    /// writing the result to `result`.
    ///
    /// ### Implementation Pseudocode
    ///
    /// ```ignore
    /// result = [0; dims]
    ///
    /// for i in range(dims):
    ///     result[i] = min(a[i], b[i])
    ///
    /// return result
    /// ```
    ///
    /// ### Result buffer
    ///
    /// The result buffer can be either an initialized slice i.e. `&mut [Self]`
    /// or it can be a slice holding potentially uninitialized data i.e. `&mut [MaybeUninit<Self>]`.
    ///
    /// Once the operation is complete, it is safe to assume the data written is fully initialized.
    ///
    /// ### Panics
    ///
    /// Panics if the size of vector `a`, `b` or `result` does not match `dims`.
    fn min_vertical<B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
    where
        B1: IntoMemLoader<Self>,
        B1::Loader: MemLoader<Value = Self>,
        B2: IntoMemLoader<Self>,
        B2::Loader: MemLoader<Value = Self>,
        for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = Self>;

    /// Checks each element pair from vectors `a` and `b` of size `dims`  comparing
    /// if element `a` is **_equal to_** element `b` returning a mask vector of the same type.
    ///
    /// ### Implementation Pseudocode
    ///
    /// ```ignore
    /// mask = [0; dims]
    ///
    /// for i in range(dims):
    ///     mask[i] = a[i] == b[i] ? 1 : 0
    ///
    /// return mask
    /// ```
    ///
    /// ### Note on `NaN` handling on `f32/f64` types
    ///
    /// For `f32` and `f64` types, `NaN` values are handled as always being `false` in **ANY** comparison.
    /// Even when compared against each other.
    ///
    /// - `0.0 == 0.0 -> true`
    /// - `0.0 == NaN -> false`
    /// - `NaN == NaN -> false`
    ///
    /// ### Result buffer
    ///
    /// The result buffer can be either an initialized slice i.e. `&mut [Self]`
    /// or it can be a slice holding potentially uninitialized data i.e. `&mut [MaybeUninit<Self>]`.
    ///
    /// Once the operation is complete, it is safe to assume the data written is fully initialized.
    ///
    /// ### Panics
    ///
    /// Panics if the size of vector `a`, `b` or `result` does not match `dims`.
    fn eq_vertical<B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
    where
        B1: IntoMemLoader<Self>,
        B1::Loader: MemLoader<Value = Self>,
        B2: IntoMemLoader<Self>,
        B2::Loader: MemLoader<Value = Self>,
        for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = Self>;

    /// Checks each element pair from vectors `a` and `b` of size `dims`  comparing
    /// if element `a` is **_not equal to_** element `b` returning a mask vector of the same type.
    ///
    /// ### Implementation Pseudocode
    ///
    /// ```ignore
    /// mask = [0; dims]
    ///
    /// for i in range(dims):
    ///     mask[i] = a[i] != b[i] ? 1 : 0
    ///
    /// return mask
    /// ```
    ///
    /// ### Note on `NaN` handling on `f32/f64` types
    ///
    /// For `f32` and `f64` types, `NaN` values are handled as always being `false` in **ANY** comparison.
    /// Even when compared against each other, meaning in the case of NOT equal, they become `true`.
    ///
    /// - `0.0 != 1.0 -> true`
    /// - `0.0 != NaN -> true`
    /// - `NaN != NaN -> true`
    ///
    /// ### Result buffer
    ///
    /// The result buffer can be either an initialized slice i.e. `&mut [Self]`
    /// or it can be a slice holding potentially uninitialized data i.e. `&mut [MaybeUninit<Self>]`.
    ///
    /// Once the operation is complete, it is safe to assume the data written is fully initialized.
    ///
    /// ### Panics
    ///
    /// Panics if the size of vector `a`, `b` or `result` does not match `dims`.
    fn neq_vertical<B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
    where
        B1: IntoMemLoader<Self>,
        B1::Loader: MemLoader<Value = Self>,
        B2: IntoMemLoader<Self>,
        B2::Loader: MemLoader<Value = Self>,
        for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = Self>;

    /// Checks each element pair from vectors `a` and `b` of size `dims`  comparing
    /// if element `a` is **_less than_** element `b` returning a mask vector of the same type.
    ///
    /// ### Implementation Pseudocode
    ///
    /// ```ignore
    /// mask = [0; dims]
    ///
    /// for i in range(dims):
    ///     mask[i] = a[i] < b[i] ? 1 : 0
    ///
    /// return mask
    /// ```
    ///
    /// ### Note on `NaN` handling on `f32/f64` types
    ///
    /// For `f32` and `f64` types, `NaN` values are handled as always being `false` in **ANY** comparison.
    /// Even when compared against each other.
    ///
    /// - `0.0 < 1.0 -> true`
    /// - `0.0 < NaN -> false`
    /// - `NaN < NaN -> false`
    ///
    /// ### Result buffer
    ///
    /// The result buffer can be either an initialized slice i.e. `&mut [Self]`
    /// or it can be a slice holding potentially uninitialized data i.e. `&mut [MaybeUninit<Self>]`.
    ///
    /// Once the operation is complete, it is safe to assume the data written is fully initialized.
    ///
    /// ### Panics
    ///
    /// Panics if the size of vector `a`, `b` or `result` does not match `dims`.
    fn lt_vertical<B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
    where
        B1: IntoMemLoader<Self>,
        B1::Loader: MemLoader<Value = Self>,
        B2: IntoMemLoader<Self>,
        B2::Loader: MemLoader<Value = Self>,
        for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = Self>;

    /// Checks each element pair from vectors `a` and `b` of size `dims`  comparing
    /// if element `a` is **_less than or equal to_** element `b` returning a mask vector of the same type.
    ///
    /// ### Implementation Pseudocode
    ///
    /// ```ignore
    /// mask = [0; dims]
    ///
    /// for i in range(dims):
    ///     mask[i] = a[i] <= b[i] ? 1 : 0
    ///
    /// return mask
    /// ```
    ///
    /// ### Note on `NaN` handling on `f32/f64` types
    ///
    /// For `f32` and `f64` types, `NaN` values are handled as always being `false` in **ANY** comparison.
    /// Even when compared against each other.
    ///
    /// - `0.0 <= 1.0 -> true`
    /// - `0.0 <= NaN -> false`
    /// - `NaN <= NaN -> false`
    ///
    /// ### Result buffer
    ///
    /// The result buffer can be either an initialized slice i.e. `&mut [Self]`
    /// or it can be a slice holding potentially uninitialized data i.e. `&mut [MaybeUninit<Self>]`.
    ///
    /// Once the operation is complete, it is safe to assume the data written is fully initialized.
    ///
    /// ### Panics
    ///
    /// Panics if the size of vector `a`, `b` or `result` does not match `dims`.
    fn lte_vertical<B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
    where
        B1: IntoMemLoader<Self>,
        B1::Loader: MemLoader<Value = Self>,
        B2: IntoMemLoader<Self>,
        B2::Loader: MemLoader<Value = Self>,
        for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = Self>;

    /// Checks each element pair from vectors `a` and `b` of size `dims`  comparing
    /// if element `a` is **_greater than_** element `b` returning a mask vector of the same type.
    ///
    /// ### Implementation Pseudocode
    ///
    /// ```ignore
    /// mask = [0; dims]
    ///
    /// for i in range(dims):
    ///     mask[i] = a[i] > b[i] ? 1 : 0
    ///
    /// return mask
    /// ```
    ///
    /// ### Note on `NaN` handling on `f32/f64` types
    ///
    /// For `f32` and `f64` types, `NaN` values are handled as always being `false` in **ANY** comparison.
    /// Even when compared against each other.
    ///
    /// - `1.0 > 0.0 -> true`
    /// - `1.0 > NaN -> false`
    /// - `NaN > NaN -> false`
    ///
    /// ### Result buffer
    ///
    /// The result buffer can be either an initialized slice i.e. `&mut [Self]`
    /// or it can be a slice holding potentially uninitialized data i.e. `&mut [MaybeUninit<Self>]`.
    ///
    /// Once the operation is complete, it is safe to assume the data written is fully initialized.
    ///
    /// ### Panics
    ///
    /// Panics if the size of vector `a`, `b` or `result` does not match `dims`.
    fn gt_vertical<B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
    where
        B1: IntoMemLoader<Self>,
        B1::Loader: MemLoader<Value = Self>,
        B2: IntoMemLoader<Self>,
        B2::Loader: MemLoader<Value = Self>,
        for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = Self>;

    /// Checks each element pair from vectors `a` and `b` of size `dims`  comparing
    /// if element `a` is **_greater than_** element `b` returning a mask vector of the same type.
    ///
    /// ### Implementation Pseudocode
    ///
    /// ```ignore
    /// mask = [0; dims]
    ///
    /// for i in range(dims):
    ///     mask[i] = a[i] >= b[i] ? 1 : 0
    ///
    /// return mask
    /// ```
    ///
    /// ### Note on `NaN` handling on `f32/f64` types
    ///
    /// For `f32` and `f64` types, `NaN` values are handled as always being `false` in **ANY** comparison.
    /// Even when compared against each other.
    ///
    /// - `1.0 >= 0.0 -> true`
    /// - `1.0 >= NaN -> false`
    /// - `NaN >= NaN -> false`
    ///
    /// ### Result buffer
    ///
    /// The result buffer can be either an initialized slice i.e. `&mut [Self]`
    /// or it can be a slice holding potentially uninitialized data i.e. `&mut [MaybeUninit<Self>]`.
    ///
    /// Once the operation is complete, it is safe to assume the data written is fully initialized.
    ///
    /// ### Panics
    ///
    /// Panics if the size of vector `a`, `b` or `result` does not match `dims`.
    fn gte_vertical<B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
    where
        B1: IntoMemLoader<Self>,
        B1::Loader: MemLoader<Value = Self>,
        B2: IntoMemLoader<Self>,
        B2::Loader: MemLoader<Value = Self>,
        for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = Self>;
}

macro_rules! cmp_ops {
    ($t:ty) => {
        impl CmpOps for $t {
            fn max<B1>(a: B1) -> Self
            where
                B1: IntoMemLoader<Self>,
                B1::Loader: MemLoader<Value = Self>,
            {
                unsafe {
                    crate::dispatch!(
                        avx512 = export_cmp_ops::generic_avx512_cmp_max,
                        avx2 = export_cmp_ops::generic_avx2_cmp_max,
                        neon = export_cmp_ops::generic_neon_cmp_max,
                        fallback = export_cmp_ops::generic_fallback_cmp_max,
                        args = (a)
                    )
                }
            }

            fn max_vertical<B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
            where
                B1: IntoMemLoader<Self>,
                B1::Loader: MemLoader<Value = Self>,
                B2: IntoMemLoader<Self>,
                B2::Loader: MemLoader<Value = Self>,
                for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = Self>,
            {
                unsafe {
                    crate::dispatch!(
                        avx512 = export_cmp_ops::generic_avx512_cmp_max_vertical,
                        avx2 = export_cmp_ops::generic_avx2_cmp_max_vertical,
                        neon = export_cmp_ops::generic_neon_cmp_max_vertical,
                        fallback = export_cmp_ops::generic_fallback_cmp_max_vertical,
                        args = (lhs, rhs, result)
                    )
                }
            }

            fn min<B1>(a: B1) -> Self
            where
                B1: IntoMemLoader<Self>,
                B1::Loader: MemLoader<Value = Self>,
            {
                unsafe {
                    crate::dispatch!(
                        avx512 = export_cmp_ops::generic_avx512_cmp_min,
                        avx2 = export_cmp_ops::generic_avx2_cmp_min,
                        neon = export_cmp_ops::generic_neon_cmp_min,
                        fallback = export_cmp_ops::generic_fallback_cmp_min,
                        args = (a)
                    )
                }
            }

            fn min_vertical<B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
            where
                B1: IntoMemLoader<Self>,
                B1::Loader: MemLoader<Value = Self>,
                B2: IntoMemLoader<Self>,
                B2::Loader: MemLoader<Value = Self>,
                for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = Self>,
            {
                unsafe {
                    crate::dispatch!(
                        avx512 = export_cmp_ops::generic_avx512_cmp_min_vertical,
                        avx2 = export_cmp_ops::generic_avx2_cmp_min_vertical,
                        neon = export_cmp_ops::generic_neon_cmp_min_vertical,
                        fallback = export_cmp_ops::generic_fallback_cmp_min_vertical,
                        args = (lhs, rhs, result)
                    )
                }
            }

            fn eq_vertical<B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
            where
                B1: IntoMemLoader<Self>,
                B1::Loader: MemLoader<Value = Self>,
                B2: IntoMemLoader<Self>,
                B2::Loader: MemLoader<Value = Self>,
                for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = Self>,
            {
                unsafe {
                    crate::dispatch!(
                        avx512 = export_cmp_ops::generic_avx512_cmp_eq_vertical,
                        avx2 = export_cmp_ops::generic_avx2_cmp_eq_vertical,
                        neon = export_cmp_ops::generic_neon_cmp_eq_vertical,
                        fallback = export_cmp_ops::generic_fallback_cmp_eq_vertical,
                        args = (lhs, rhs, result)
                    )
                }
            }

            fn neq_vertical<B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
            where
                B1: IntoMemLoader<Self>,
                B1::Loader: MemLoader<Value = Self>,
                B2: IntoMemLoader<Self>,
                B2::Loader: MemLoader<Value = Self>,
                for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = Self>,
            {
                unsafe {
                    crate::dispatch!(
                        avx512 = export_cmp_ops::generic_avx512_cmp_neq_vertical,
                        avx2 = export_cmp_ops::generic_avx2_cmp_neq_vertical,
                        neon = export_cmp_ops::generic_neon_cmp_neq_vertical,
                        fallback = export_cmp_ops::generic_fallback_cmp_neq_vertical,
                        args = (lhs, rhs, result)
                    )
                }
            }

            fn lt_vertical<B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
            where
                B1: IntoMemLoader<Self>,
                B1::Loader: MemLoader<Value = Self>,
                B2: IntoMemLoader<Self>,
                B2::Loader: MemLoader<Value = Self>,
                for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = Self>,
            {
                unsafe {
                    crate::dispatch!(
                        avx512 = export_cmp_ops::generic_avx512_cmp_lt_vertical,
                        avx2 = export_cmp_ops::generic_avx2_cmp_lt_vertical,
                        neon = export_cmp_ops::generic_neon_cmp_lt_vertical,
                        fallback = export_cmp_ops::generic_fallback_cmp_lt_vertical,
                        args = (lhs, rhs, result)
                    )
                }
            }

            fn lte_vertical<B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
            where
                B1: IntoMemLoader<Self>,
                B1::Loader: MemLoader<Value = Self>,
                B2: IntoMemLoader<Self>,
                B2::Loader: MemLoader<Value = Self>,
                for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = Self>,
            {
                unsafe {
                    crate::dispatch!(
                        avx512 = export_cmp_ops::generic_avx512_cmp_lte_vertical,
                        avx2 = export_cmp_ops::generic_avx2_cmp_lte_vertical,
                        neon = export_cmp_ops::generic_neon_cmp_lte_vertical,
                        fallback = export_cmp_ops::generic_fallback_cmp_lte_vertical,
                        args = (lhs, rhs, result)
                    )
                }
            }

            fn gt_vertical<B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
            where
                B1: IntoMemLoader<Self>,
                B1::Loader: MemLoader<Value = Self>,
                B2: IntoMemLoader<Self>,
                B2::Loader: MemLoader<Value = Self>,
                for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = Self>,
            {
                unsafe {
                    crate::dispatch!(
                        avx512 = export_cmp_ops::generic_avx512_cmp_gt_vertical,
                        avx2 = export_cmp_ops::generic_avx2_cmp_gt_vertical,
                        neon = export_cmp_ops::generic_neon_cmp_gt_vertical,
                        fallback = export_cmp_ops::generic_fallback_cmp_gt_vertical,
                        args = (lhs, rhs, result)
                    )
                }
            }

            fn gte_vertical<B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
            where
                B1: IntoMemLoader<Self>,
                B1::Loader: MemLoader<Value = Self>,
                B2: IntoMemLoader<Self>,
                B2::Loader: MemLoader<Value = Self>,
                for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = Self>,
            {
                unsafe {
                    crate::dispatch!(
                        avx512 = export_cmp_ops::generic_avx512_cmp_gte_vertical,
                        avx2 = export_cmp_ops::generic_avx2_cmp_gte_vertical,
                        neon = export_cmp_ops::generic_neon_cmp_gte_vertical,
                        fallback = export_cmp_ops::generic_fallback_cmp_gte_vertical,
                        args = (lhs, rhs, result)
                    )
                }
            }
        }
    };
}

cmp_ops!(f32);
cmp_ops!(f64);
cmp_ops!(i8);
cmp_ops!(i16);
cmp_ops!(i32);
cmp_ops!(i64);
cmp_ops!(u8);
cmp_ops!(u16);
cmp_ops!(u32);
cmp_ops!(u64);
