use core::ops::Neg;

use cfavml::danger::{DenseLane, SimdRegister};
use cfavml::math::{AutoMath, Math};
use num_complex::Complex;
use num_traits::Float;

use super::complex_ops::ComplexOps;
use crate::math::{AutoComplex, ComplexMath};

pub struct Fallback;

impl<T> ComplexOps<T> for Fallback
where
    T: Copy + Neg<Output = T>,
    AutoMath: Math<T>,
{
    type Register = Complex<T>;

    type HalfRegister = T;

    unsafe fn conj(value: Self::Register) -> Self::Register {
        Complex {
            re: value.re,
            im: value.im.neg(),
        }
    }

    unsafe fn inv(value: Self::Register) -> Self::Register {
        let squared_norm = AutoMath::add(
            AutoMath::mul(value.re, value.re),
            AutoMath::mul(value.im, value.im),
        );
        let conj = <Fallback as ComplexOps<T>>::conj(value);

        Complex {
            re: AutoMath::div(conj.re, squared_norm),
            im: AutoMath::div(conj.im, squared_norm),
        }
    }
}

impl<T> SimdRegister<Complex<T>> for Fallback
where
    T: Float,
    AutoMath: Math<T>,
    AutoComplex: ComplexMath<T>,
{
    type Register = Complex<T>;

    #[inline(always)]
    unsafe fn load(mem: *const Complex<T>) -> Self::Register {
        mem.read()
    }

    #[inline(always)]
    unsafe fn filled(value: Complex<T>) -> Self::Register {
        value
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        AutoComplex::zero()
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        AutoComplex::add(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        AutoComplex::sub(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        AutoComplex::mul(l1, l2)
    }

    #[inline(always)]
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        AutoComplex::div(l1, l2)
    }

    #[inline(always)]
    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        let res = AutoComplex::mul(l1, l2);
        AutoComplex::add(res, acc)
    }

    #[inline(always)]
    unsafe fn fmadd_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
        acc: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let res = <Self as SimdRegister<Complex<T>>>::mul_dense(l1, l2);
        <Self as SimdRegister<Complex<T>>>::add_dense(res, acc)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut Complex<T>, reg: Self::Register) {
        mem.write(reg)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        Complex {
            re: AutoMath::cast_bool(AutoMath::cmp_eq(l1.re, l2.re)),
            im: AutoMath::cast_bool(AutoMath::cmp_eq(l1.im, l2.im)),
        }
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        Complex {
            re: AutoMath::cast_bool(!AutoMath::cmp_eq(l1.re, l2.re)),
            im: AutoMath::cast_bool(!AutoMath::cmp_eq(l1.im, l2.im)),
        }
    }

    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> Complex<T> {
        reg
    }

    unsafe fn max(_l1: Self::Register, _l2: Self::Register) -> Self::Register {
        unimplemented!("Ordering operations are not supported for complex numbers");
    }

    unsafe fn min(_l1: Self::Register, _l22: Self::Register) -> Self::Register {
        unimplemented!("Ordering operations are not supported for complex numbers");
    }

    unsafe fn lt(_l1: Self::Register, _l2: Self::Register) -> Self::Register {
        unimplemented!("Ordering operations are not supported for complex numbers");
    }

    unsafe fn lte(_l1: Self::Register, _l2: Self::Register) -> Self::Register {
        unimplemented!("Ordering operations are not supported for complex numbers");
    }

    unsafe fn gt(_l1: Self::Register, _l2: Self::Register) -> Self::Register {
        unimplemented!("Ordering operations are not supported for complex numbers");
    }

    unsafe fn gte(_l1: Self::Register, _l2: Self::Register) -> Self::Register {
        unimplemented!("Ordering operations are not supported for complex numbers");
    }

    unsafe fn max_to_value(_reg: Self::Register) -> Complex<T> {
        unimplemented!("Ordering operations are not supported for complex numbers");
    }

    unsafe fn min_to_value(_reg: Self::Register) -> Complex<T> {
        unimplemented!("Ordering operations are not supported for complex numbers");
    }
}
