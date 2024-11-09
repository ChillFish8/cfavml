#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use cfavml::danger::SimdRegister;
use num_complex::Complex;

use super::complex_ops::ComplexSimdOps;
use crate::danger::complex_ops::ComplexOps;
use crate::danger::impl_avx2::Avx2Complex;
pub struct Avx2ComplexFma;

impl ComplexSimdOps<f32> for Avx2ComplexFma {
    type Register = __m256;
    unsafe fn dup_real_components(value: Self::Register) -> Self::Register {
        _mm256_moveldup_ps(value)
    }

    unsafe fn dup_imag_components(value: Self::Register) -> Self::Register {
        _mm256_movehdup_ps(value)
    }

    unsafe fn dup_norm(value: Self::Register) -> Self::Register {
        let (real, imag) =
            <Avx2Complex as ComplexSimdOps<f32>>::dup_complex_components(value);
        let b_abs = _mm256_andnot_ps(imag, _mm256_set1_ps(-0.0));
        let ab = _mm256_div_ps(real, imag);
        let ab_square = _mm256_mul_ps(ab, ab);
        _mm256_mul_ps(
            b_abs,
            _mm256_sqrt_ps(_mm256_fmadd_ps(ab_square, ab_square, _mm256_set1_ps(1.0))),
        )
    }

    unsafe fn swap_complex_components(value: Self::Register) -> Self::Register {
        <Avx2Complex as ComplexSimdOps<f32>>::swap_complex_components(value)
    }
}

impl ComplexOps<f32> for Avx2ComplexFma {
    type Register = __m256;
    type HalfRegister = __m128;

    #[inline(always)]
    unsafe fn conj(value: Self::Register) -> Self::Register {
        <Avx2Complex as ComplexOps<f32>>::conj(value)
    }
    #[inline(always)]
    unsafe fn inv(value: Self::Register) -> Self::Register {
        let norm = <Avx2ComplexFma as ComplexSimdOps<f32>>::dup_norm(value);
        _mm256_div_ps(
            _mm256_div_ps(<Avx2Complex as ComplexOps<f32>>::conj(value), norm),
            norm,
        )
    }
}

impl SimdRegister<Complex<f32>> for Avx2ComplexFma {
    type Register = __m256;

    #[inline(always)]
    unsafe fn mul(left: Self::Register, right: Self::Register) -> Self::Register {
        let (left_real, left_imag) =
            <Avx2Complex as ComplexSimdOps<f32>>::dup_complex_components(left);

        let right_shuffled =
            <Avx2Complex as ComplexSimdOps<f32>>::swap_complex_components(right);

        let output_right = _mm256_mul_ps(left_imag, right_shuffled);

        _mm256_fmaddsub_ps(left_real, right, output_right)
    }

    #[inline(always)]
    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        // complex multiplication is already requires an fmadd, so we can't use the intrinsic
        // directly
        _mm256_add_ps(
            <Avx2ComplexFma as SimdRegister<Complex<f32>>>::mul(l1, l2),
            acc,
        )
    }

    #[inline(always)]
    unsafe fn load(mem: *const Complex<f32>) -> Self::Register {
        <Avx2Complex as SimdRegister<Complex<f32>>>::load(mem)
    }

    #[inline(always)]
    unsafe fn filled(value: Complex<f32>) -> Self::Register {
        <Avx2Complex as SimdRegister<Complex<f32>>>::filled(value)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        <Avx2Complex as SimdRegister<Complex<f32>>>::zeroed()
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Avx2Complex as SimdRegister<Complex<f32>>>::add(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Avx2Complex as SimdRegister<Complex<f32>>>::sub(l1, l2)
    }

    #[inline(always)]
    unsafe fn div(left: Self::Register, right: Self::Register) -> Self::Register {
        let right = <Avx2ComplexFma as ComplexOps<f32>>::inv(right);
        <Avx2ComplexFma as SimdRegister<Complex<f32>>>::mul(left, right)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Avx2Complex as SimdRegister<Complex<f32>>>::eq(l1, l2)
    }
    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Avx2Complex as SimdRegister<Complex<f32>>>::neq(l1, l2)
    }
    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> Complex<f32> {
        <Avx2Complex as SimdRegister<Complex<f32>>>::sum_to_value(reg)
    }

    unsafe fn write(mem: *mut Complex<f32>, reg: Self::Register) {
        <Avx2Complex as SimdRegister<Complex<f32>>>::write(mem, reg)
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

    unsafe fn max_to_value(_reg: Self::Register) -> Complex<f32> {
        unimplemented!("Ordering operations are not supported for complex numbers");
    }

    unsafe fn min_to_value(_reg: Self::Register) -> Complex<f32> {
        unimplemented!("Ordering operations are not supported for complex numbers");
    }
}

impl ComplexSimdOps<f64> for Avx2ComplexFma {
    type Register = __m256d;
    unsafe fn dup_real_components(value: Self::Register) -> Self::Register {
        <Avx2Complex as ComplexSimdOps<f64>>::dup_real_components(value)
    }

    unsafe fn dup_imag_components(value: Self::Register) -> Self::Register {
        <Avx2Complex as ComplexSimdOps<f64>>::dup_imag_components(value)
    }

    unsafe fn dup_norm(value: Self::Register) -> Self::Register {
        let (real, imag) =
            <Avx2Complex as ComplexSimdOps<f64>>::dup_complex_components(value);
        let b_abs = _mm256_andnot_pd(imag, _mm256_set1_pd(-0.0));
        let ab = _mm256_div_pd(real, imag);
        let ab_square = _mm256_mul_pd(ab, ab);
        _mm256_mul_pd(
            b_abs,
            _mm256_sqrt_pd(_mm256_fmadd_pd(ab_square, ab_square, _mm256_set1_pd(1.0))),
        )
    }

    unsafe fn swap_complex_components(value: Self::Register) -> Self::Register {
        <Avx2Complex as ComplexSimdOps<f64>>::swap_complex_components(value)
    }
}
impl ComplexOps<f64> for Avx2ComplexFma {
    type Register = __m256d;
    type HalfRegister = __m128d;

    #[inline(always)]
    unsafe fn conj(value: Self::Register) -> Self::Register {
        <Avx2Complex as ComplexOps<f64>>::conj(value)
    }
    #[inline(always)]
    unsafe fn inv(value: Self::Register) -> Self::Register {
        let norm = <Avx2ComplexFma as ComplexSimdOps<f64>>::dup_norm(value);
        _mm256_div_pd(
            _mm256_div_pd(<Avx2Complex as ComplexOps<f64>>::conj(value), norm),
            norm,
        )
    }
}

impl SimdRegister<Complex<f64>> for Avx2ComplexFma {
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
            <Avx2Complex as ComplexSimdOps<f64>>::dup_complex_components(left);
        let right_shuffled =
            <Avx2Complex as ComplexSimdOps<f64>>::swap_complex_components(right);
        let output_right = _mm256_mul_pd(left_imag, right_shuffled);
        _mm256_fmaddsub_pd(left_real, right, output_right)
    }

    #[inline(always)]
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let l2_inv = <Avx2ComplexFma as ComplexOps<f64>>::inv(l2);
        <Avx2ComplexFma as SimdRegister<Complex<f64>>>::mul(l1, l2_inv)
    }

    #[inline(always)]
    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        <Avx2ComplexFma as SimdRegister<Complex<f64>>>::add(
            <Avx2ComplexFma as SimdRegister<Complex<f64>>>::mul(l1, l2),
            acc,
        )
    }

    unsafe fn max(_l1: Self::Register, _l2: Self::Register) -> Self::Register {
        unimplemented!("Ordering operations are not supported for complex numbers");
    }

    unsafe fn min(_l1: Self::Register, _l2: Self::Register) -> Self::Register {
        unimplemented!("Ordering operations are not supported for complex numbers");
    }

    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Avx2Complex as SimdRegister<Complex<f64>>>::eq(l1, l2)
    }

    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Avx2Complex as SimdRegister<Complex<f64>>>::neq(l1, l2)
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

    unsafe fn sum_to_value(reg: Self::Register) -> Complex<f64> {
        <Avx2Complex as SimdRegister<Complex<f64>>>::sum_to_value(reg)
    }

    unsafe fn max_to_value(_reg: Self::Register) -> Complex<f64> {
        unimplemented!("Ordering operations are not supported for complex numbers");
    }

    unsafe fn min_to_value(_reg: Self::Register) -> Complex<f64> {
        unimplemented!("Ordering operations are not supported for complex numbers");
    }

    unsafe fn write(mem: *mut Complex<f64>, reg: Self::Register) {
        <Avx2Complex as SimdRegister<Complex<f64>>>::write(mem, reg)
    }
}
