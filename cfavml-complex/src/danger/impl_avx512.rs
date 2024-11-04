use crate::danger::complex_ops::ComplexOps;
use cfavml::danger::SimdRegister;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use num_complex::Complex;

pub struct Avx512Complex;

impl ComplexOps<f32> for Avx512Complex {
    type Register = __m512;
    type HalfRegister = __m256;

    unsafe fn duplicate_reals(value: Self::Register) -> Self::Register {
        todo!()
    }

    unsafe fn imag_values(value: Self::Register) -> Self::Register {
        todo!()
    }

    unsafe fn conj(value: Self::Register) -> Self::Register {
        todo!()
    }

    unsafe fn inv(value: Self::Register) -> Self::Register {
        todo!()
    }

    unsafe fn dup_norm(value: Self::Register) -> Self::Register {
        todo!()
    }

    unsafe fn swap_complex_components(value: Self::Register) -> Self::Register {
        todo!()
    }
}
