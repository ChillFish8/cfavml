//! Safe but somewhat low-level variants of the arithmetic operations in CFAVML.
//!
//! In general, I would recommend using the higher level generic functions api which provides
//! some syntax sugar over these traits.

use crate::buffer::WriteOnlyBuffer;
use crate::danger::export_arithmetic_ops;
use crate::mem_loader::{IntoMemLoader, MemLoader};

/// Various arithmetic operations over vectors.
pub trait ArithmeticOps: Sized + Copy {
    /// Performs an element wise addition of two input buffers `lhs` and `rhs` that can
    /// be projected to the desired output size of `result`.
    ///
    /// ### Projecting Vectors
    ///
    /// CFAVML allows for working over a wide variety of buffers for applications, projection is effectively
    /// broadcasting of two input buffers implementing `IntoMemLoader<T>`.
    ///
    /// By default, you can provide _two slices_, _one slice and a broadcast value_, or _two broadcast values_,
    /// which exhibit the standard behaviour as you might expect.
    ///
    /// When providing two slices as inputs they cannot be projected to a buffer
    /// that is larger their input sizes by default. This means providing two slices
    /// of `128` elements in length must take a result buffer of `128` elements in length.
    ///
    /// ### Implementation Pseudocode
    ///
    /// ```ignore
    /// result = [0; dims]
    ///
    /// for i in range(dims):
    ///     result[i] = a[i] + b[i]
    ///
    /// return result
    /// ```
    ///
    /// # Panics
    ///
    /// If vectors `a` and `b` cannot be projected to the target size of `result`.
    /// Note that the projection rules are tied to the `MemLoader` implementation.
    fn add_vertical<B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
    where
        B1: IntoMemLoader<Self>,
        B1::Loader: MemLoader<Value = Self>,
        B2: IntoMemLoader<Self>,
        B2::Loader: MemLoader<Value = Self>,
        for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = Self>;

    /// Performs an element wise subtraction of two input buffers `a` and `b` that can
    /// be projected to the desired output size of `result`.
    ///
    /// ### Projecting Vectors
    ///
    /// CFAVML allows for working over a wide variety of buffers for applications, projection is effectively
    /// broadcasting of two input buffers implementing `IntoMemLoader<T>`.
    ///
    /// By default, you can provide _two slices_, _one slice and a broadcast value_, or _two broadcast values_,
    /// which exhibit the standard behaviour as you might expect.
    ///
    /// When providing two slices as inputs they cannot be projected to a buffer
    /// that is larger their input sizes by default. This means providing two slices
    /// of `128` elements in length must take a result buffer of `128` elements in length.
    ///
    /// ### Implementation Pseudocode
    ///
    /// ```ignore
    /// result = [0; dims]
    ///
    /// for i in range(dims):
    ///     result[i] = a[i] - b[i]
    ///
    /// return result
    /// ```
    ///
    /// # Panics
    ///
    /// If vectors `a` and `b` cannot be projected to the target size of `result`.
    /// Note that the projection rules are tied to the `MemLoader` implementation.
    fn sub_vertical<B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
    where
        B1: IntoMemLoader<Self>,
        B1::Loader: MemLoader<Value = Self>,
        B2: IntoMemLoader<Self>,
        B2::Loader: MemLoader<Value = Self>,
        for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = Self>;

    /// Performs an element wise multiply of two input buffers `a` and `b` that can
    /// be projected to the desired output size of `result`.
    ///
    /// ### Projecting Vectors
    ///
    /// CFAVML allows for working over a wide variety of buffers for applications, projection is effectively
    /// broadcasting of two input buffers implementing `IntoMemLoader<T>`.
    ///
    /// By default, you can provide _two slices_, _one slice and a broadcast value_, or _two broadcast values_,
    /// which exhibit the standard behaviour as you might expect.
    ///
    /// When providing two slices as inputs they cannot be projected to a buffer
    /// that is larger their input sizes by default. This means providing two slices
    /// of `128` elements in length must take a result buffer of `128` elements in length.
    ///
    /// ### Implementation Pseudocode
    ///
    /// ```ignore
    /// result = [0; dims]
    ///
    /// for i in range(dims):
    ///     result[i] = a[i] * b[i]
    ///
    /// return result
    /// ```
    ///
    /// # Panics
    ///
    /// If vectors `a` and `b` cannot be projected to the target size of `result`.
    /// Note that the projection rules are tied to the `MemLoader` implementation.
    fn mul_vertical<B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
    where
        B1: IntoMemLoader<Self>,
        B1::Loader: MemLoader<Value = Self>,
        B2: IntoMemLoader<Self>,
        B2::Loader: MemLoader<Value = Self>,
        for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = Self>;

    /// Performs an element wise division of two input buffers `a` and `b` that can
    /// be projected to the desired output size of `result`.
    ///
    /// ### Projecting Vectors
    ///
    /// CFAVML allows for working over a wide variety of buffers for applications, projection is effectively
    /// broadcasting of two input buffers implementing `IntoMemLoader<T>`.
    ///
    /// By default, you can provide _two slices_, _one slice and a broadcast value_, or _two broadcast values_,
    /// which exhibit the standard behaviour as you might expect.
    ///
    /// When providing two slices as inputs they cannot be projected to a buffer
    /// that is larger their input sizes by default. This means providing two slices
    /// of `128` elements in length must take a result buffer of `128` elements in length.
    ///
    /// ### Implementation Pseudocode
    ///
    /// ```ignore
    /// result = [0; dims]
    ///
    /// for i in range(dims):
    ///     result[i] = a[i] / b[i]
    ///
    /// return result
    /// ```
    ///
    /// # Panics
    ///
    /// If vectors `a` and `b` cannot be projected to the target size of `result`.
    /// Note that the projection rules are tied to the `MemLoader` implementation.
    fn div_vertical<B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
    where
        B1: IntoMemLoader<Self>,
        B1::Loader: MemLoader<Value = Self>,
        B2: IntoMemLoader<Self>,
        B2::Loader: MemLoader<Value = Self>,
        for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = Self>;
}

macro_rules! arithmetic_ops {
    ($t:ty) => {
        impl ArithmeticOps for $t {
            fn add_vertical<B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
            where
                B1: IntoMemLoader<Self>,
                B1::Loader: MemLoader<Value = Self>,
                B2: IntoMemLoader<Self>,
                B2::Loader: MemLoader<Value = Self>,
                for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = Self>,
            {
                unsafe {
                    crate::dispatch!(
                        avx512 = export_arithmetic_ops::generic_avx512_add_vertical,
                        avx2 = export_arithmetic_ops::generic_avx2_add_vertical,
                        neon = export_arithmetic_ops::generic_neon_add_vertical,
                        fallback = export_arithmetic_ops::generic_fallback_add_vertical,
                        args = (lhs, rhs, result)
                    );
                }
            }

            fn sub_vertical<B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
            where
                B1: IntoMemLoader<Self>,
                B1::Loader: MemLoader<Value = Self>,
                B2: IntoMemLoader<Self>,
                B2::Loader: MemLoader<Value = Self>,
                for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = Self>,
            {
                unsafe {
                    crate::dispatch!(
                        avx512 = export_arithmetic_ops::generic_avx512_sub_vertical,
                        avx2 = export_arithmetic_ops::generic_avx2_sub_vertical,
                        neon = export_arithmetic_ops::generic_neon_sub_vertical,
                        fallback = export_arithmetic_ops::generic_fallback_sub_vertical,
                        args = (lhs, rhs, result)
                    );
                }
            }

            fn mul_vertical<B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
            where
                B1: IntoMemLoader<Self>,
                B1::Loader: MemLoader<Value = Self>,
                B2: IntoMemLoader<Self>,
                B2::Loader: MemLoader<Value = Self>,
                for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = Self>,
            {
                unsafe {
                    crate::dispatch!(
                        avx512 = export_arithmetic_ops::generic_avx512_mul_vertical,
                        avx2 = export_arithmetic_ops::generic_avx2_mul_vertical,
                        neon = export_arithmetic_ops::generic_neon_mul_vertical,
                        fallback = export_arithmetic_ops::generic_fallback_mul_vertical,
                        args = (lhs, rhs, result)
                    );
                }
            }

            fn div_vertical<B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
            where
                B1: IntoMemLoader<Self>,
                B1::Loader: MemLoader<Value = Self>,
                B2: IntoMemLoader<Self>,
                B2::Loader: MemLoader<Value = Self>,
                for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = Self>,
            {
                unsafe {
                    crate::dispatch!(
                        avx512 = export_arithmetic_ops::generic_avx512_div_vertical,
                        avx2 = export_arithmetic_ops::generic_avx2_div_vertical,
                        neon = export_arithmetic_ops::generic_neon_div_vertical,
                        fallback = export_arithmetic_ops::generic_fallback_div_vertical,
                        args = (lhs, rhs, result)
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
