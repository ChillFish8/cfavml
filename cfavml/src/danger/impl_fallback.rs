use crate::danger::{DenseLane, SimdRegister};
use crate::math::{AutoMath, Math};

/// Fallback SIMD-like operations.
///
/// This is designed to provide abstract operations that are easily optimized by the compiler
/// even if we're not manually writing the SIMD, hopefully to cover other architectures that
/// we haven't manually supported.
pub struct Fallback;

impl<T> SimdRegister<T> for Fallback
where
    T: Copy,
    AutoMath: Math<T>,
{
    type Register = T;

    #[inline(always)]
    unsafe fn load(mem: *const T) -> Self::Register {
        mem.read()
    }

    #[inline(always)]
    unsafe fn filled(value: T) -> Self::Register {
        value
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        AutoMath::zero()
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        AutoMath::add(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        AutoMath::sub(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        AutoMath::mul(l1, l2)
    }

    #[inline(always)]
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        AutoMath::div(l1, l2)
    }

    #[inline(always)]
    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        let res = AutoMath::mul(l1, l2);
        AutoMath::add(res, acc)
    }

    #[inline(always)]
    unsafe fn fmadd_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
        acc: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let res = <Self as SimdRegister<T>>::mul_dense(l1, l2);
        <Self as SimdRegister<T>>::add_dense(res, acc)
    }

    #[inline(always)]
    unsafe fn max(l1: Self::Register, l2: Self::Register) -> Self::Register {
        AutoMath::cmp_max(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        AutoMath::cmp_min(l1, l2)
    }

    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> T {
        reg
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> T {
        reg
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> T {
        reg
    }

    #[inline(always)]
    unsafe fn write(mem: *mut T, reg: Self::Register) {
        mem.write(reg)
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        to_mask(AutoMath::cmp_lt(l1, l2))
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        to_mask(AutoMath::cmp_lte(l1, l2))
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        to_mask(AutoMath::cmp_gt(l1, l2))
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        to_mask(AutoMath::cmp_gte(l1, l2))
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        to_mask(AutoMath::cmp_eq(l1, l2))
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        to_mask(!AutoMath::cmp_eq(l1, l2))
    }
}

#[inline(always)]
fn to_mask<T>(v: bool) -> T
where
    AutoMath: Math<T>,
{
    if v {
        AutoMath::one()
    } else {
        AutoMath::zero()
    }
}
