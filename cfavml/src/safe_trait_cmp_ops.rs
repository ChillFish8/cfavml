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
    fn max_horizontal(dims: usize, a: &[Self]) -> Self;

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
    fn min_horizontal(dims: usize, a: &[Self]) -> Self;

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
}

macro_rules! cmp_ops {
    ($t:ty) => {
        impl CmpOps for $t {
            fn max_horizontal(dims: usize, a: &[Self]) -> Self {
                assert_eq!(a.len(), dims, "Input vector `a` does not match size `dims`");

                unsafe {
                    crate::dispatch!(
                        avx512 = export_cmp_ops::generic_avx512_max_horizontal,
                        avx2 = export_cmp_ops::generic_avx2_max_horizontal,
                        neon = export_cmp_ops::generic_neon_max_horizontal,
                        fallback = export_cmp_ops::generic_fallback_max_horizontal,
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
                        avx512 = export_cmp_ops::generic_avx512_max_value,
                        avx2 = export_cmp_ops::generic_avx2_max_value,
                        neon = export_cmp_ops::generic_neon_max_value,
                        fallback = export_cmp_ops::generic_fallback_max_value,
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
                        avx512 = export_cmp_ops::generic_avx512_max_vector,
                        avx2 = export_cmp_ops::generic_avx2_max_vector,
                        neon = export_cmp_ops::generic_neon_max_vector,
                        fallback = export_cmp_ops::generic_fallback_max_vector,
                        args = (dims, a, b, result)
                    )
                }
            }

            fn min_horizontal(dims: usize, a: &[Self]) -> Self {
                assert_eq!(a.len(), dims, "Input vector `a` does not match size `dims`");

                unsafe {
                    crate::dispatch!(
                        avx512 = export_cmp_ops::generic_avx512_min_horizontal,
                        avx2 = export_cmp_ops::generic_avx2_min_horizontal,
                        neon = export_cmp_ops::generic_neon_min_horizontal,
                        fallback = export_cmp_ops::generic_fallback_min_horizontal,
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
                        avx512 = export_cmp_ops::generic_avx512_min_value,
                        avx2 = export_cmp_ops::generic_avx2_min_value,
                        neon = export_cmp_ops::generic_neon_min_value,
                        fallback = export_cmp_ops::generic_fallback_min_value,
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
                        avx512 = export_cmp_ops::generic_avx512_min_vector,
                        avx2 = export_cmp_ops::generic_avx2_min_vector,
                        neon = export_cmp_ops::generic_neon_min_vector,
                        fallback = export_cmp_ops::generic_fallback_min_vector,
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
