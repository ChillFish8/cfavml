//! Safe but somewhat low-level variants of the aggregation operations in CFAVML.
//!
//! In general, I would recommend using the higher level generic functions api which provides
//! some syntax sugar over these traits.

use crate::danger::export_agg_ops;
use crate::mem_loader::{IntoMemLoader, MemLoader};

/// Various aggregation operations on a single vector.
pub trait AggOps: Sized {
    /// Performs a horizontal sum of all elements in `a` returning the result.
    ///
    /// ### Pseudocode
    ///
    /// ```ignore
    /// result = 0
    ///
    /// for i in range(dims):
    ///     result += a[i]
    ///
    /// return result
    /// ```
    fn sum<B1>(a: B1) -> Self
    where
        B1: IntoMemLoader<Self>,
        B1::Loader: MemLoader<Value = Self>;
}

macro_rules! agg_ops {
    ($t:ty) => {
        impl AggOps for $t {
            fn sum<B1>(a: B1) -> Self
            where
                B1: IntoMemLoader<Self>,
                B1::Loader: MemLoader<Value = Self>,
            {
                unsafe {
                    crate::dispatch!(
                        avx512 = export_agg_ops::generic_avx512_sum,
                        avx2 = export_agg_ops::generic_avx2_sum,
                        neon = export_agg_ops::generic_neon_sum,
                        fallback = export_agg_ops::generic_fallback_sum,
                        args = (a)
                    )
                }
            }
        }
    };
}

agg_ops!(f32);
agg_ops!(f64);
agg_ops!(i8);
agg_ops!(i16);
agg_ops!(i32);
agg_ops!(i64);
agg_ops!(u8);
agg_ops!(u16);
agg_ops!(u32);
agg_ops!(u64);
