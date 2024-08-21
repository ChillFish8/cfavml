//! Safe but somewhat low-level variants of the comparison operations in CFAVML.
//!
//! In general, I would recommend using the higher level generic functions api which provides
//! some syntax sugar over these traits.

use crate::buffer::WriteOnlyBuffer;
use crate::danger::export_cmp_ops;

/// Various comparison operations over vectors.
pub trait CmpOps: Sized {
    /// Finds the horizontal max element of a given vector and returns the result.
    ///
    /// ### Pseudocode
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
    fn max(dims: usize, a: &[Self]) -> Self;

    /// Performs an element wise max on each element of vector `a` and the provided broadcast
    /// value, writing the result to `result`.
    ///
    /// ### Pseudocode
    ///
    /// ```ignore
    /// result = [0; dims]
    ///
    /// for i in range(dims):
    ///     result[i] = max(value, a[i])
    ///
    /// return result
    /// ```
    ///
    /// ### Result buffer
    ///
    /// The result buffer can be either an initialized slice i.e. [`&mut [Self]`]
    /// or it can be a slice holding potentially uninitialized data i.e. [`&mut [MaybeUninit<Self>]`].
    ///
    /// Once the operation is complete, it is safe to assume the data written is fully initialized.
    ///
    /// ### Panics
    ///
    /// Panics if the size of vector `a` or `result` does not match `dims`.
    fn max_value<B>(dims: usize, value: Self, a: &[Self], result: &mut [B])
    where
        for<'a> &'a mut [B]: WriteOnlyBuffer<Item = Self>;

    /// Performs an element wise max on each element of vector `a` and `b`,
    /// writing the result to `result`.
    ///
    /// ### Pseudocode
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
    /// The result buffer can be either an initialized slice i.e. [`&mut [Self]`]
    /// or it can be a slice holding potentially uninitialized data i.e. [`&mut [MaybeUninit<Self>]`].
    ///
    /// Once the operation is complete, it is safe to assume the data written is fully initialized.
    ///
    /// ### Panics
    ///
    /// Panics if the size of vector `a`, `b` or `result` does not match `dims`.
    fn max_vector<B>(dims: usize, a: &[Self], b: &[Self], result: &mut [B])
    where
        for<'a> &'a mut [B]: WriteOnlyBuffer<Item = Self>;

    /// Finds the horizontal min element of a given vector.
    ///
    /// ### Pseudocode
    ///
    /// ```ignore
    /// result = inf
    ///
    /// for i in range(dims):
    ///     result = min(result, a[i])
    ///
    /// return result
    /// ```
    fn min(dims: usize, a: &[Self]) -> Self;

    /// Performs an element wise min on each element of vector `a` and the provided broadcast
    /// value, writing the result to `result`.
    ///
    /// ### Pseudocode
    ///
    /// ```ignore
    /// result = [0; dims]
    ///
    /// for i in range(dims):
    ///     result[i] = min(value, a[i])
    ///
    /// return result
    /// ```
    ///
    /// ### Result buffer
    ///
    /// The result buffer can be either an initialized slice i.e. [`&mut [Self]`]
    /// or it can be a slice holding potentially uninitialized data i.e. [`&mut [MaybeUninit<Self>]`].
    ///
    /// Once the operation is complete, it is safe to assume the data written is fully initialized.
    ///
    /// ### Panics
    ///
    /// Panics if the size of vector `a` or `result` does not match `dims`.
    fn min_value<B>(dims: usize, value: Self, a: &[Self], result: &mut [B])
    where
        for<'a> &'a mut [B]: WriteOnlyBuffer<Item = Self>;

    /// Performs an element wise min on each element of vector `a` and `b`,
    /// writing the result to `result`.
    ///
    /// ### Pseudocode
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
    /// The result buffer can be either an initialized slice i.e. [`&mut [Self]`]
    /// or it can be a slice holding potentially uninitialized data i.e. [`&mut [MaybeUninit<Self>]`].
    ///
    /// Once the operation is complete, it is safe to assume the data written is fully initialized.
    ///
    /// ### Panics
    ///
    /// Panics if the size of vector `a`, `b` or `result` does not match `dims`.
    fn min_vector<B>(dims: usize, a: &[Self], b: &[Self], result: &mut [B])
    where
        for<'a> &'a mut [B]: WriteOnlyBuffer<Item = Self>;

    /// Checks each element within vector `a` of size `dims` against a provided broadcast value
    /// comparing if they are **_equal_** returning a mask vector of the same type.
    ///
    /// ### Pseudocode
    ///
    /// ```ignore
    /// mask = [0; dims]
    ///
    /// for i in range(dims):
    /// mask[i] = a[i] == value ? 1 : 0
    ///
    /// return mask
    /// ```
    ///
    /// ### Note on `Nan` handling on `f32/f64` types
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
    /// The result buffer can be either an initialized slice i.e. [`&mut [Self]`]
    /// or it can be a slice holding potentially uninitialized data i.e. [`&mut [MaybeUninit<Self>]`].
    ///
    /// Once the operation is complete, it is safe to assume the data written is fully initialized.
    ///
    /// ### Panics
    ///
    /// Panics if the size of vector `a` or `result` does not match `dims`.
    fn eq_value<B>(dims: usize, value: Self, a: &[Self], result: &mut [B])
    where
        for<'a> &'a mut [B]: WriteOnlyBuffer<Item = Self>;

    /// Checks each element pair from vectors `a` and `b` of size `dims`  comparing
    /// if element `a` is **_equal to_** element `b` returning a mask vector of the same type.
    ///
    /// ### Pseudocode
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
    /// ### Note on `Nan` handling on `f32/f64` types
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
    /// The result buffer can be either an initialized slice i.e. [`&mut [Self]`]
    /// or it can be a slice holding potentially uninitialized data i.e. [`&mut [MaybeUninit<Self>]`].
    ///
    /// Once the operation is complete, it is safe to assume the data written is fully initialized.
    ///
    /// ### Panics
    ///
    /// Panics if the size of vector `a`, `b` or `result` does not match `dims`.
    fn eq_vector<B>(dims: usize, a: &[Self], b: &[Self], result: &mut [B])
    where
        for<'a> &'a mut [B]: WriteOnlyBuffer<Item = Self>;

    /// Checks each element within vector `a` of size `dims` against a provided broadcast value
    /// comparing if they are **_not equal_** returning a mask vector of the same type.
    ///
    /// ### Pseudocode
    ///
    /// ```ignore
    /// mask = [0; dims]
    ///
    /// for i in range(dims):
    /// mask[i] = a[i] != value ? 1 : 0
    ///
    /// return mask
    /// ```
    ///
    /// ### Note on `Nan` handling on `f32/f64` types
    ///
    /// For `f32` and `f64` types, `NaN` values are handled as always being `false` in **ANY** comparison.
    /// Even when compared against each other.
    ///
    /// - `0.0 != 1.0 -> true`
    /// - `0.0 != NaN -> false`
    /// - `NaN != NaN -> false`
    ///
    /// ### Result buffer
    ///
    /// The result buffer can be either an initialized slice i.e. [`&mut [Self]`]
    /// or it can be a slice holding potentially uninitialized data i.e. [`&mut [MaybeUninit<Self>]`].
    ///
    /// Once the operation is complete, it is safe to assume the data written is fully initialized.
    ///
    /// ### Panics
    ///
    /// Panics if the size of vector `a` or `result` does not match `dims`.
    fn neq_value<B>(dims: usize, value: Self, a: &[Self], result: &mut [B])
    where
        for<'a> &'a mut [B]: WriteOnlyBuffer<Item = Self>;

    /// Checks each element pair from vectors `a` and `b` of size `dims`  comparing
    /// if element `a` is **_not equal to_** element `b` returning a mask vector of the same type.
    ///
    /// ### Pseudocode
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
    /// ### Note on `Nan` handling on `f32/f64` types
    ///
    /// For `f32` and `f64` types, `NaN` values are handled as always being `false` in **ANY** comparison.
    /// Even when compared against each other.
    ///
    /// - `0.0 != 1.0 -> true`
    /// - `0.0 != NaN -> false`
    /// - `NaN != NaN -> false`
    ///
    /// ### Result buffer
    ///
    /// The result buffer can be either an initialized slice i.e. [`&mut [Self]`]
    /// or it can be a slice holding potentially uninitialized data i.e. [`&mut [MaybeUninit<Self>]`].
    ///
    /// Once the operation is complete, it is safe to assume the data written is fully initialized.
    ///
    /// ### Panics
    ///
    /// Panics if the size of vector `a`, `b` or `result` does not match `dims`.
    fn neq_vector<B>(dims: usize, a: &[Self], b: &[Self], result: &mut [B])
    where
        for<'a> &'a mut [B]: WriteOnlyBuffer<Item = Self>;

    /// Checks each element within vector `a` of size `dims` against a provided broadcast value
    /// comparing if they are **_less than_** returning a mask vector of the same type.
    ///
    /// ### Pseudocode
    ///
    /// ```ignore
    /// mask = [0; dims]
    ///
    /// for i in range(dims):
    /// mask[i] = a[i] < value ? 1 : 0
    ///
    /// return mask
    /// ```
    ///
    /// ### Note on `Nan` handling on `f32/f64` types
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
    /// The result buffer can be either an initialized slice i.e. [`&mut [Self]`]
    /// or it can be a slice holding potentially uninitialized data i.e. [`&mut [MaybeUninit<Self>]`].
    ///
    /// Once the operation is complete, it is safe to assume the data written is fully initialized.
    ///
    /// ### Panics
    ///
    /// Panics if the size of vector `a` or `result` does not match `dims`.
    fn lt_value<B>(dims: usize, value: Self, a: &[Self], result: &mut [B])
    where
        for<'a> &'a mut [B]: WriteOnlyBuffer<Item = Self>;

    /// Checks each element pair from vectors `a` and `b` of size `dims`  comparing
    /// if element `a` is **_less than_** element `b` returning a mask vector of the same type.
    ///
    /// ### Pseudocode
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
    /// ### Note on `Nan` handling on `f32/f64` types
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
    /// The result buffer can be either an initialized slice i.e. [`&mut [Self]`]
    /// or it can be a slice holding potentially uninitialized data i.e. [`&mut [MaybeUninit<Self>]`].
    ///
    /// Once the operation is complete, it is safe to assume the data written is fully initialized.
    ///
    /// ### Panics
    ///
    /// Panics if the size of vector `a`, `b` or `result` does not match `dims`.
    fn lt_vector<B>(dims: usize, a: &[Self], b: &[Self], result: &mut [B])
    where
        for<'a> &'a mut [B]: WriteOnlyBuffer<Item = Self>;

    /// Checks each element within vector `a` of size `dims` against a provided broadcast value
    /// comparing if they are **_less than or equal_** returning a mask vector of the same type.
    ///
    /// ### Pseudocode
    ///
    /// ```ignore
    /// mask = [0; dims]
    ///
    /// for i in range(dims):
    /// mask[i] = a[i] <= value ? 1 : 0
    ///
    /// return mask
    /// ```
    ///
    /// ### Note on `Nan` handling on `f32/f64` types
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
    /// The result buffer can be either an initialized slice i.e. [`&mut [Self]`]
    /// or it can be a slice holding potentially uninitialized data i.e. [`&mut [MaybeUninit<Self>]`].
    ///
    /// Once the operation is complete, it is safe to assume the data written is fully initialized.
    ///
    /// ### Panics
    ///
    /// Panics if the size of vector `a` or `result` does not match `dims`.
    fn lte_value<B>(dims: usize, value: Self, a: &[Self], result: &mut [B])
    where
        for<'a> &'a mut [B]: WriteOnlyBuffer<Item = Self>;

    /// Checks each element pair from vectors `a` and `b` of size `dims`  comparing
    /// if element `a` is **_less than or equal to_** element `b` returning a mask vector of the same type.
    ///
    /// ### Pseudocode
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
    /// ### Note on `Nan` handling on `f32/f64` types
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
    /// The result buffer can be either an initialized slice i.e. [`&mut [Self]`]
    /// or it can be a slice holding potentially uninitialized data i.e. [`&mut [MaybeUninit<Self>]`].
    ///
    /// Once the operation is complete, it is safe to assume the data written is fully initialized.
    ///
    /// ### Panics
    ///
    /// Panics if the size of vector `a`, `b` or `result` does not match `dims`.
    fn lte_vector<B>(dims: usize, a: &[Self], b: &[Self], result: &mut [B])
    where
        for<'a> &'a mut [B]: WriteOnlyBuffer<Item = Self>;

    /// Checks each element within vector `a` of size `dims` against a provided broadcast value
    /// comparing if they are **_less than or equal_** returning a mask vector of the same type.
    ///
    /// ### Pseudocode
    ///
    /// ```ignore
    /// mask = [0; dims]
    ///
    /// for i in range(dims):
    /// mask[i] = a[i] > value ? 1 : 0
    ///
    /// return mask
    /// ```
    ///
    /// ### Note on `Nan` handling on `f32/f64` types
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
    /// The result buffer can be either an initialized slice i.e. [`&mut [Self]`]
    /// or it can be a slice holding potentially uninitialized data i.e. [`&mut [MaybeUninit<Self>]`].
    ///
    /// Once the operation is complete, it is safe to assume the data written is fully initialized.
    ///
    /// ### Panics
    ///
    /// Panics if the size of vector `a` or `result` does not match `dims`.
    fn gt_value<B>(dims: usize, value: Self, a: &[Self], result: &mut [B])
    where
        for<'a> &'a mut [B]: WriteOnlyBuffer<Item = Self>;

    /// Checks each element pair from vectors `a` and `b` of size `dims`  comparing
    /// if element `a` is **_greater than_** element `b` returning a mask vector of the same type.
    ///
    /// ### Pseudocode
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
    /// ### Note on `Nan` handling on `f32/f64` types
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
    /// The result buffer can be either an initialized slice i.e. [`&mut [Self]`]
    /// or it can be a slice holding potentially uninitialized data i.e. [`&mut [MaybeUninit<Self>]`].
    ///
    /// Once the operation is complete, it is safe to assume the data written is fully initialized.
    ///
    /// ### Panics
    ///
    /// Panics if the size of vector `a`, `b` or `result` does not match `dims`.
    fn gt_vector<B>(dims: usize, a: &[Self], b: &[Self], result: &mut [B])
    where
        for<'a> &'a mut [B]: WriteOnlyBuffer<Item = Self>;

    /// Checks each element within vector `a` of size `dims` against a provided broadcast value
    /// comparing if they are **_less than or equal_** returning a mask vector of the same type.
    ///
    /// ### Pseudocode
    ///
    /// ```ignore
    /// mask = [0; dims]
    ///
    /// for i in range(dims):
    /// mask[i] = a[i] >= value ? 1 : 0
    ///
    /// return mask
    /// ```
    ///
    /// ### Note on `Nan` handling on `f32/f64` types
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
    /// The result buffer can be either an initialized slice i.e. [`&mut [Self]`]
    /// or it can be a slice holding potentially uninitialized data i.e. [`&mut [MaybeUninit<Self>]`].
    ///
    /// Once the operation is complete, it is safe to assume the data written is fully initialized.
    ///
    /// ### Panics
    ///
    /// Panics if the size of vector `a` or `result` does not match `dims`.
    fn gte_value<B>(dims: usize, value: Self, a: &[Self], result: &mut [B])
    where
        for<'a> &'a mut [B]: WriteOnlyBuffer<Item = Self>;

    /// Checks each element pair from vectors `a` and `b` of size `dims`  comparing
    /// if element `a` is **_greater than_** element `b` returning a mask vector of the same type.
    ///
    /// ### Pseudocode
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
    /// ### Note on `Nan` handling on `f32/f64` types
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
    /// The result buffer can be either an initialized slice i.e. [`&mut [Self]`]
    /// or it can be a slice holding potentially uninitialized data i.e. [`&mut [MaybeUninit<Self>]`].
    ///
    /// Once the operation is complete, it is safe to assume the data written is fully initialized.
    ///
    /// ### Panics
    ///
    /// Panics if the size of vector `a`, `b` or `result` does not match `dims`.
    fn gte_vector<B>(dims: usize, a: &[Self], b: &[Self], result: &mut [B])
    where
        for<'a> &'a mut [B]: WriteOnlyBuffer<Item = Self>;
}

macro_rules! cmp_ops {
    ($t:ty) => {
        impl CmpOps for $t {
            fn max(dims: usize, a: &[Self]) -> Self {
                assert_eq!(a.len(), dims, "Input vector `a` does not match size `dims`");

                unsafe {
                    crate::dispatch!(
                        avx512 = export_cmp_ops::generic_avx512_cmp_max,
                        avx2 = export_cmp_ops::generic_avx2_cmp_max,
                        neon = export_cmp_ops::generic_neon_cmp_max,
                        fallback = export_cmp_ops::generic_fallback_cmp_max,
                        args = (dims, a)
                    )
                }
            }

            fn max_value<B>(dims: usize, value: Self, a: &[Self], result: &mut [B])
            where
                for<'a> &'a mut [B]: WriteOnlyBuffer<Item = Self>,
            {
                assert_eq!(a.len(), dims, "Input vector `a` does not match size `dims`");
                assert_eq!(
                    result.len(),
                    dims,
                    "Input vector `result` does not match size `dims`"
                );

                unsafe {
                    crate::dispatch!(
                        avx512 = export_cmp_ops::generic_avx512_cmp_max_value,
                        avx2 = export_cmp_ops::generic_avx2_cmp_max_value,
                        neon = export_cmp_ops::generic_neon_cmp_max_value,
                        fallback = export_cmp_ops::generic_fallback_cmp_max_value,
                        args = (dims, value, a, result)
                    )
                }
            }

            fn max_vector<B>(dims: usize, a: &[Self], b: &[Self], result: &mut [B])
            where
                for<'a> &'a mut [B]: WriteOnlyBuffer<Item = Self>,
            {
                assert_eq!(a.len(), dims, "Input vector `a` does not match size `dims`");
                assert_eq!(b.len(), dims, "Input vector `b` does not match size `dims`");
                assert_eq!(
                    result.len(),
                    dims,
                    "Input vector `result` does not match size `dims`"
                );

                unsafe {
                    crate::dispatch!(
                        avx512 = export_cmp_ops::generic_avx512_cmp_max_vector,
                        avx2 = export_cmp_ops::generic_avx2_cmp_max_vector,
                        neon = export_cmp_ops::generic_neon_cmp_max_vector,
                        fallback = export_cmp_ops::generic_fallback_cmp_max_vector,
                        args = (dims, a, b, result)
                    )
                }
            }

            fn min(dims: usize, a: &[Self]) -> Self {
                assert_eq!(a.len(), dims, "Input vector `a` does not match size `dims`");

                unsafe {
                    crate::dispatch!(
                        avx512 = export_cmp_ops::generic_avx512_cmp_min,
                        avx2 = export_cmp_ops::generic_avx2_cmp_min,
                        neon = export_cmp_ops::generic_neon_cmp_min,
                        fallback = export_cmp_ops::generic_fallback_cmp_min,
                        args = (dims, a)
                    )
                }
            }

            fn min_value<B>(dims: usize, value: Self, a: &[Self], result: &mut [B])
            where
                for<'a> &'a mut [B]: WriteOnlyBuffer<Item = Self>,
            {
                assert_eq!(a.len(), dims, "Input vector `a` does not match size `dims`");
                assert_eq!(
                    result.len(),
                    dims,
                    "Input vector `result` does not match size `dims`"
                );

                unsafe {
                    crate::dispatch!(
                        avx512 = export_cmp_ops::generic_avx512_cmp_min_value,
                        avx2 = export_cmp_ops::generic_avx2_cmp_min_value,
                        neon = export_cmp_ops::generic_neon_cmp_min_value,
                        fallback = export_cmp_ops::generic_fallback_cmp_min_value,
                        args = (dims, value, a, result)
                    )
                }
            }

            fn min_vector<B>(dims: usize, a: &[Self], b: &[Self], result: &mut [B])
            where
                for<'a> &'a mut [B]: WriteOnlyBuffer<Item = Self>,
            {
                assert_eq!(a.len(), dims, "Input vector `a` does not match size `dims`");
                assert_eq!(b.len(), dims, "Input vector `b` does not match size `dims`");
                assert_eq!(
                    result.len(),
                    dims,
                    "Input vector `result` does not match size `dims`"
                );

                unsafe {
                    crate::dispatch!(
                        avx512 = export_cmp_ops::generic_avx512_cmp_min_vector,
                        avx2 = export_cmp_ops::generic_avx2_cmp_min_vector,
                        neon = export_cmp_ops::generic_neon_cmp_min_vector,
                        fallback = export_cmp_ops::generic_fallback_cmp_min_vector,
                        args = (dims, a, b, result)
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
