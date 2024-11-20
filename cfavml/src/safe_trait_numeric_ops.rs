//! Safe but somewhat low-level variants of the arithmetic operations in CFAVML.
//!
//! In general, I would recommend using the higher level generic functions api which provides
//! some syntax sugar over these traits.

use crate::buffer::WriteOnlyBuffer;
use crate::danger::export_hypot;
use crate::mem_loader::{IntoMemLoader, MemLoader};

/// Various arithmetic operations over vectors.
pub trait NumericOps: Sized + Copy {
    /// Performs an elementwise hypotenuse of input buffers `a` and `b` that can
    /// be projected to the desired output size of `result`. Implementation is an appoximation that
    /// _should_ match std::hypot in most cases. However, with some inputs it's been confirmed to be off by 1 ulp.
    ///
    /// See [cfavml::hypot_vertical](crate::hypot_vertical) for examples.
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
    /// You can wrap your inputs in a [Projected](crate::mem_loader::Projected) wrapper which
    /// enables projecting of the input buffer to new sizes providing the new size is a
    /// multiple of the original size. When this buffer is projected, it is effectively
    /// repeated `N` times, where `N` is how many times the old size fits into the new size.
    ///
    /// ### Implementation Pseudocode
    ///
    /// ```ignore
    /// result = [0; dims]
    ///
    /// for i in range(dims):
    ///     result[i] = a[i].hypot(b[i])
    ///
    /// return result
    /// ```
    ///
    /// # Panics
    ///
    /// If vectors `a` and `b` cannot be projected to the target size of `result`.
    /// Note that the projection rules are tied to the `MemLoader` implementation.
    fn hypot_vertical<B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
    where
        B1: IntoMemLoader<Self>,
        B1::Loader: MemLoader<Value = Self>,
        B2: IntoMemLoader<Self>,
        B2::Loader: MemLoader<Value = Self>,
        for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = Self>;
}

macro_rules! numeric_ops {
    ($t:ty) => {
        impl NumericOps for $t {
            fn hypot_vertical<B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
            where
                B1: IntoMemLoader<Self>,
                B1::Loader: MemLoader<Value = Self>,
                B2: IntoMemLoader<Self>,
                B2::Loader: MemLoader<Value = Self>,
                for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = Self>,
            {
                unsafe {
                    crate::dispatch!(
                        avx512 = export_hypot::generic_avx512_hypot_vertical,
                        avx2 = export_hypot::generic_avx2_hypot_vertical,
                        neon = export_hypot::generic_neon_hypot_vertical,
                        fallback = export_hypot::generic_fallback_hypot_vertical,
                        args = (lhs, rhs, result)
                    );
                }
            }
        }
    };
}

numeric_ops!(f32);
numeric_ops!(f64);
