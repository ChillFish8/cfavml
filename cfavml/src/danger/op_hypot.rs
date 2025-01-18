use super::core_routine_boilerplate::apply_vertical_kernel;
use super::core_simd_api::Hypot;
use crate::buffer::WriteOnlyBuffer;
use crate::danger::core_simd_api::SimdRegister;
use crate::math::{Math, Numeric};
use crate::mem_loader::{IntoMemLoader, MemLoader};

#[inline(always)]
/// A generic vector implementation of hypot over one vector and single value.
///
/// # Safety
///
/// The safety of hypot relies on the safety of the implementation of SimdRegister
/// The sizes of `a`, `b` and `result` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.

pub unsafe fn generic_hypot_vertical<T, R, M, B1, B2, B3>(
    a: B1,
    b: B2,
    result: &mut [B3],
) where
    T: Copy,
    R: SimdRegister<T> + Hypot<T>,
    M: Math<T> + Numeric<T>,
    B1: IntoMemLoader<T>,
    B1::Loader: MemLoader<Value = T>,
    B2: IntoMemLoader<T>,
    B2::Loader: MemLoader<Value = T>,
    for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = T>,
{
    apply_vertical_kernel::<T, R, M, B1, B2, B3>(
        a,
        b,
        result,
        R::hypot_dense,
        R::hypot,
        M::hypot,
    )
}
