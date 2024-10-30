#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use cfavml::danger::{Avx2, SimdRegister};
use num_complex::Complex;
pub trait ComplexOps<T>
where
    Complex<T>: Copy,
{
    type Register: Copy;
    /// Extracts the real component from a register
    unsafe fn real_values(value: Self::Register) -> Self::Register;
    /// Extracts the imaginary component from a register
    unsafe fn imag_values(value: Self::Register) -> Self::Register;

    /// Duplicates the real and imaginary components of a the input into separate registers
    unsafe fn duplicate_complex_components(
        value: Self::Register,
    ) -> (Self::Register, Self::Register) {
        (Self::real_values(value), Self::imag_values(value))
    }
    /// Swaps the real and imaginary components of a complex number
    unsafe fn swap_complex_components(value: Self::Register) -> Self::Register;
}

pub struct Avx2Complex;
impl ComplexOps<f32> for Avx2Complex {
    type Register = __m256;

    #[inline(always)]
    unsafe fn real_values(value: Self::Register) -> Self::Register {
        _mm256_moveldup_ps(value)
    }

    #[inline(always)]
    unsafe fn imag_values(value: Self::Register) -> Self::Register {
        _mm256_movehdup_ps(value)
    }
    #[inline(always)]
    unsafe fn swap_complex_components(value: Self::Register) -> Self::Register {
        _mm256_permute_ps(value, 0xB1)
    }
}

impl SimdRegister<Complex<f32>> for Avx2Complex {
    type Register = __m256;

    #[inline(always)]
    unsafe fn load(mem: *const Complex<f32>) -> Self::Register {
        _mm256_loadu_ps(mem as *const f32)
    }

    #[inline(always)]
    unsafe fn filled(value: Complex<f32>) -> Self::Register {
        // no
        _mm256_setr_ps(
            value.re, value.im, value.re, value.im, value.re, value.im, value.re,
            value.im,
        )
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        _mm256_setzero_ps()
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_add_ps(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_sub_ps(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(left: Self::Register, right: Self::Register) -> Self::Register {
        // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        // [a, b, a, b, a, b, a, b], [c, d, c, d, c, d, c, d]
        // [a x 4], [b x 4]
        let (left_real, left_imag) =
            <Avx2Complex as ComplexOps<f32>>::duplicate_complex_components(left);
        let right_shuffled =
            <Avx2Complex as ComplexOps<f32>>::swap_complex_components(right);

        let output_right = _mm256_mul_ps(left_imag, right_shuffled);
        _mm256_fmaddsub_ps(left_real, right, output_right)
    }

    #[inline(always)]
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        todo!()
    }

    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        todo!()
    }

    unsafe fn max(l1: Self::Register, l2: Self::Register) -> Self::Register {
        todo!()
    }

    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        todo!()
    }

    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        todo!()
    }

    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        todo!()
    }

    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        todo!()
    }

    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        todo!()
    }

    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        todo!()
    }

    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        todo!()
    }

    unsafe fn sum_to_value(reg: Self::Register) -> Complex<f32> {
        todo!()
    }

    unsafe fn max_to_value(reg: Self::Register) -> Complex<f32> {
        todo!()
    }

    unsafe fn min_to_value(reg: Self::Register) -> Complex<f32> {
        todo!()
    }

    unsafe fn write(mem: *mut Complex<f32>, reg: Self::Register) {
        todo!()
    }
}

impl ComplexOps<Complex<f64>> for Avx2Complex {
    type Register = __m256d;
    #[inline(always)]
    unsafe fn real_values(value: Self::Register) -> Self::Register {
        _mm256_movedup_pd(value)
    }
    #[inline(always)]
    unsafe fn imag_values(value: Self::Register) -> Self::Register {
        _mm256_permute_pd(value, 0x0F)
    }
    #[inline(always)]
    unsafe fn swap_complex_components(value: Self::Register) -> Self::Register {
        _mm256_permute_pd(value, 0x5)
    }
}

impl SimdRegister<Complex<f64>> for Avx2Complex {
    type Register = __m256d;

    #[inline(always)]
    unsafe fn load(mem: *const Complex<f64>) -> Self::Register {
        _mm256_loadu_pd(mem as *const f64)
    }

    #[inline(always)]
    unsafe fn filled(value: Complex<f64>) -> Self::Register {
        _mm256_setr_pd(value.re, value.im, value.re, value.im)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        _mm256_setzero_pd()
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_add_pd(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_sub_pd(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(left: Self::Register, right: Self::Register) -> Self::Register {
        let (left_real, left_imag) =
            (_mm256_movedup_pd(left), _mm256_permute_pd(left, 0x0F));
        let right_shuffled = _mm256_permute_pd(right, 0x5);
        let output_right = _mm256_mul_pd(left_imag, right_shuffled);
        _mm256_fmaddsub_pd(left_real, right, output_right)
    }

    #[inline(always)]
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        todo!()
    }

    #[inline(always)]
    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        todo!()
    }

    unsafe fn max(l1: Self::Register, l2: Self::Register) -> Self::Register {
        todo!()
    }

    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        todo!()
    }

    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        todo!()
    }

    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        todo!()
    }

    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        todo!()
    }

    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        todo!()
    }

    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        todo!()
    }

    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        todo!()
    }

    unsafe fn sum_to_value(reg: Self::Register) -> Complex<f64> {
        todo!()
    }

    unsafe fn max_to_value(reg: Self::Register) -> Complex<f64> {
        todo!()
    }

    unsafe fn min_to_value(reg: Self::Register) -> Complex<f64> {
        todo!()
    }

    unsafe fn write(mem: *mut Complex<f64>, reg: Self::Register) {
        todo!()
    }
}
