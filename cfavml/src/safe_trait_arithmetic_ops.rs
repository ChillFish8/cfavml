//! Safe but somewhat low-level variants of the arithmetic operations in CFAVML.
//!
//! In general, I would recommend using the higher level generic functions api which provides
//! some syntax sugar over these traits.

use crate::buffer::WriteOnlyBuffer;
use crate::danger::export_arithmetic_ops;

/// Various arithmetic operations over vectors.
pub trait ArithmeticOps: Sized {
    /// Performs an element wise addition of each element of vector `a` and the provided broadcast
    /// value, writing the result to `result`.
    ///
    /// ### Pseudocode
    ///
    /// ```py
    /// result = [0; dims]
    ///
    /// for i in range(dims):
    ///     result[i] = a[i] + value
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
    /// Panics if the size of vectors `a` or `result` does not match `dims`.
    fn add_value<B>(dims: usize, value: Self, a: &[Self], result: &mut [B])
    where
        for<'a> &'a mut [B]: WriteOnlyBuffer<Item = Self>;

    /// Performs an element wise addition of each element pair of vector `a` and `b`,
    /// writing the result to `result`.
    ///
    /// ### Pseudocode
    ///
    /// ```py
    /// result = [0; dims]
    ///
    /// for i in range(dims):
    ///     result[i] = a[i] + b[i]
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
    /// Panics if the size of vectors `a`, `b` or `result` does not match `dims`.
    fn add_vector<B>(dims: usize, a: &[Self], b: &[Self], result: &mut [B])
    where
        for<'a> &'a mut [B]: WriteOnlyBuffer<Item = Self>;

    /// Performs an element wise subtraction of each element of vector `a` and the provided broadcast
    /// value, writing the result to `result`.
    ///
    /// ### Pseudocode
    ///
    /// ```py
    /// result = [0; dims]
    ///
    /// for i in range(dims):
    ///     result[i] = a[i] - value
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
    /// Panics if the size of vectors `a` or `result` does not match `dims`.
    fn sub_value<B>(dims: usize, value: Self, a: &[Self], result: &mut [B])
    where
        for<'a> &'a mut [B]: WriteOnlyBuffer<Item = Self>;

    /// Performs an element wise subtraction of each element pair from vectors `a` and `b`,
    /// writing the result to `result`.
    ///
    /// ### Pseudocode
    ///
    /// ```py
    /// result = [0; dims]
    ///
    /// for i in range(dims):
    ///     result[i] = a[i] - b[i]
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
    /// Panics if the size of vectors `a`, `b` or `result` does not match `dims`.
    fn sub_vector<B>(dims: usize, a: &[Self], b: &[Self], result: &mut [B])
    where
        for<'a> &'a mut [B]: WriteOnlyBuffer<Item = Self>;

    /// Performs an element wise multiplication of each element of vector `a` and the provided broadcast
    /// value, writing the result to `result`.
    ///
    /// ### Pseudocode
    ///
    /// ```py
    /// result = [0; dims]
    ///
    /// for i in range(dims):
    ///     result[i] = a[i] * value
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
    /// Panics if the size of vectors `a` or `result` does not match `dims`.
    fn mul_value<B>(dims: usize, value: Self, a: &[Self], result: &mut [B])
    where
        for<'a> &'a mut [B]: WriteOnlyBuffer<Item = Self>;

    /// Performs an element wise multiplication of each element pair from vectors `a` and `b`,
    /// writing the result to `result`.
    ///
    /// ### Pseudocode
    ///
    /// ```py
    /// result = [0; dims]
    ///
    /// for i in range(dims):
    ///     result[i] = a[i] * b[i]
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
    /// Panics if the size of vectors `a`, `b` or `result` does not match `dims`.
    fn mul_vector<B>(dims: usize, a: &[Self], b: &[Self], result: &mut [B])
    where
        for<'a> &'a mut [B]: WriteOnlyBuffer<Item = Self>;

    /// Performs an element wise division of each element of vector `a` and the provided broadcast
    /// value, writing the result to `result`.
    ///
    /// ### Pseudocode
    ///
    /// ```py
    /// result = [0; dims]
    ///
    /// for i in range(dims):
    ///     result[i] = a[i] / value
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
    /// Panics if the size of vectors `a` or `result` does not match `dims`.
    fn div_value<B>(dims: usize, value: Self, a: &[Self], result: &mut [B])
    where
        for<'a> &'a mut [B]: WriteOnlyBuffer<Item = Self>;

    /// Performs an element wise division on each element pair from vectors `a` and `b`,
    /// writing the result to `result`.
    ///
    /// ### Pseudocode
    ///
    /// ```py
    /// result = [0; dims]
    ///
    /// for i in range(dims):
    ///     result[i] = a[i] / b[i]
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
    /// Panics if the size of vectors `a`, `b` or `result` does not match `dims`.
    fn div_vector<B>(dims: usize, a: &[Self], b: &[Self], result: &mut [B])
    where
        for<'a> &'a mut [B]: WriteOnlyBuffer<Item = Self>;
}

macro_rules! arithmetic_ops {
    ($t:ty) => {
        impl ArithmeticOps for $t {
            fn add_value<B>(dims: usize, value: Self, a: &[Self], result: &mut [B])
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
                        avx512 = export_arithmetic_ops::generic_avx512_add_value,
                        avx2 = export_arithmetic_ops::generic_avx2_add_value,
                        neon = export_arithmetic_ops::generic_neon_add_value,
                        fallback = export_arithmetic_ops::generic_fallback_add_value,
                        args = (dims, value, a, result)
                    );
                }
            }

            fn add_vector<B>(dims: usize, a: &[Self], b: &[Self], result: &mut [B])
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
                        avx512 = export_arithmetic_ops::generic_avx512_add_vector,
                        avx2 = export_arithmetic_ops::generic_avx2_add_vector,
                        neon = export_arithmetic_ops::generic_neon_add_vector,
                        fallback = export_arithmetic_ops::generic_fallback_add_vector,
                        args = (dims, a, b, result)
                    );
                }
            }

            fn sub_value<B>(dims: usize, value: Self, a: &[Self], result: &mut [B])
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
                        avx512 = export_arithmetic_ops::generic_avx512_sub_value,
                        avx2 = export_arithmetic_ops::generic_avx2_sub_value,
                        neon = export_arithmetic_ops::generic_neon_sub_value,
                        fallback = export_arithmetic_ops::generic_fallback_sub_value,
                        args = (dims, value, a, result)
                    );
                }
            }

            fn sub_vector<B>(dims: usize, a: &[Self], b: &[Self], result: &mut [B])
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
                        avx512 = export_arithmetic_ops::generic_avx512_sub_vector,
                        avx2 = export_arithmetic_ops::generic_avx2_sub_vector,
                        neon = export_arithmetic_ops::generic_neon_sub_vector,
                        fallback = export_arithmetic_ops::generic_fallback_sub_vector,
                        args = (dims, a, b, result)
                    );
                }
            }

            fn mul_value<B>(dims: usize, value: Self, a: &[Self], result: &mut [B])
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
                        avx512 = export_arithmetic_ops::generic_avx512_mul_value,
                        avx2 = export_arithmetic_ops::generic_avx2_mul_value,
                        neon = export_arithmetic_ops::generic_neon_mul_value,
                        fallback = export_arithmetic_ops::generic_fallback_mul_value,
                        args = (dims, value, a, result)
                    );
                }
            }

            fn mul_vector<B>(dims: usize, a: &[Self], b: &[Self], result: &mut [B])
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
                        avx512 = export_arithmetic_ops::generic_avx512_mul_vector,
                        avx2 = export_arithmetic_ops::generic_avx2_mul_vector,
                        neon = export_arithmetic_ops::generic_neon_mul_vector,
                        fallback = export_arithmetic_ops::generic_fallback_mul_vector,
                        args = (dims, a, b, result)
                    );
                }
            }

            fn div_value<B>(dims: usize, value: Self, a: &[Self], result: &mut [B])
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
                        avx512 = export_arithmetic_ops::generic_avx512_div_value,
                        avx2 = export_arithmetic_ops::generic_avx2_div_value,
                        neon = export_arithmetic_ops::generic_neon_div_value,
                        fallback = export_arithmetic_ops::generic_fallback_div_value,
                        args = (dims, value, a, result)
                    );
                }
            }

            fn div_vector<B>(dims: usize, a: &[Self], b: &[Self], result: &mut [B])
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
                        avx512 = export_arithmetic_ops::generic_avx512_div_vector,
                        avx2 = export_arithmetic_ops::generic_avx2_div_vector,
                        neon = export_arithmetic_ops::generic_neon_div_vector,
                        fallback = export_arithmetic_ops::generic_fallback_div_vector,
                        args = (dims, a, b, result)
                    );
                }
            }
        }
    };
}

arithmetic_ops!(f32);
arithmetic_ops!(f64);
arithmetic_ops!(i8);
arithmetic_ops!(i16);
arithmetic_ops!(i32);
arithmetic_ops!(i64);
arithmetic_ops!(u8);
arithmetic_ops!(u16);
arithmetic_ops!(u32);
arithmetic_ops!(u64);
