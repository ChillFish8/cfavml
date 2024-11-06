#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use cfavml::danger::{Avx2, SimdRegister};
use num_complex::Complex;

use super::complex_ops::ComplexOps;

pub struct Avx2Complex;
impl ComplexOps<f32> for Avx2Complex {
    type Register = __m256;
    type HalfRegister = __m128;
    #[inline(always)]
    unsafe fn dup_real_components(value: Self::Register) -> Self::Register {
        _mm256_moveldup_ps(value)
    }

    #[inline(always)]
    unsafe fn dup_imag_components(value: Self::Register) -> Self::Register {
        _mm256_movehdup_ps(value)
    }
    #[inline(always)]
    unsafe fn swap_complex_components(value: Self::Register) -> Self::Register {
        _mm256_permute_ps(value, 0xB1)
    }
    #[inline(always)]
    unsafe fn conj(value: Self::Register) -> Self::Register {
        _mm256_xor_ps(
            value,
            _mm256_setr_ps(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0),
        )
    }

    #[inline(always)]
    unsafe fn inv(value: Self::Register) -> Self::Register {
        let norm = <Avx2Complex as ComplexOps<f32>>::dup_norm(value);
        let conj = <Avx2Complex as ComplexOps<f32>>::conj(value);
        _mm256_div_ps(_mm256_div_ps(conj, norm), norm)
    }
    #[inline(always)]
    unsafe fn dup_norm(value: Self::Register) -> Self::Register {
        let (real, imag) =
            <Avx2Complex as ComplexOps<f32>>::dup_complex_components(value);
        _mm256_sqrt_ps(_mm256_add_ps(
            _mm256_mul_ps(real, real),
            _mm256_mul_ps(imag, imag),
        ))
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
        _mm256_setr_ps(
            value.im, value.re, value.im, value.re, value.im, value.re, value.im,
            value.re,
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
        let (left_real, left_imag) =
            <Avx2Complex as ComplexOps<f32>>::dup_complex_components(left);

        let right_shuffled =
            <Avx2Complex as ComplexOps<f32>>::swap_complex_components(right);

        let output_right = _mm256_mul_ps(left_imag, right_shuffled);

        let output_left = _mm256_mul_ps(left_real, right);
        _mm256_addsub_ps(output_left, output_right)
    }

    #[inline(always)]
    unsafe fn div(left: Self::Register, right: Self::Register) -> Self::Register {
        let right = <Avx2Complex as ComplexOps<f32>>::inv(right);
        <Avx2Complex as SimdRegister<Complex<f32>>>::mul(left, right)
    }

    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        <Avx2Complex as SimdRegister<Complex<f32>>>::add(
            <Avx2Complex as SimdRegister<Complex<f32>>>::mul(l1, l2),
            acc,
        )
    }

    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = <Avx2 as SimdRegister<f32>>::eq(l1, l2);
        _mm256_and_ps(mask, _mm256_permute_ps(mask, 0xB1))
    }

    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = <Avx2 as SimdRegister<f32>>::neq(l1, l2);
        _mm256_andnot_ps(mask, _mm256_permute_ps(mask, 0xB1))
    }

    unsafe fn sum_to_value(reg: Self::Register) -> Complex<f32> {
        let (right_half, left_half) =
            (_mm256_castps256_ps128(reg), _mm256_extractf128_ps::<1>(reg));

        let sum = _mm_add_ps(left_half, right_half);
        let shuffled_sum = _mm_castps_pd(sum);
        let shuffled_sum = _mm_unpackhi_pd(shuffled_sum, shuffled_sum);
        let shuffled_sum = _mm_castpd_ps(shuffled_sum);
        let sum = _mm_add_ps(sum, shuffled_sum);
        let mut store = [Complex::new(0.0, 0.0); 1];
        _mm_store_pd(store.as_mut_ptr() as *mut f64, _mm_castps_pd(sum));
        store[0]
    }

    unsafe fn write(mem: *mut Complex<f32>, reg: Self::Register) {
        _mm256_storeu_pd(mem as *mut f64, _mm256_castps_pd(reg));
    }

    unsafe fn max(_l1: Self::Register, _l22: Self::Register) -> Self::Register {
        unimplemented!("Ordering operations are not supported for complex numbers");
    }

    unsafe fn min(_l1: Self::Register, _l2: Self::Register) -> Self::Register {
        unimplemented!("Ordering operations are not supported for complex numbers");
    }

    unsafe fn lt(_l1: Self::Register, _l22: Self::Register) -> Self::Register {
        unimplemented!("Ordering operations are not supported for complex numbers");
    }

    unsafe fn lte(_l1: Self::Register, _l2: Self::Register) -> Self::Register {
        unimplemented!("Ordering operations are not supported for complex numbers");
    }

    unsafe fn gt(_l1: Self::Register, _l22: Self::Register) -> Self::Register {
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

impl ComplexOps<f64> for Avx2Complex {
    type Register = __m256d;
    type HalfRegister = __m128d;
    #[inline(always)]
    unsafe fn dup_real_components(value: Self::Register) -> Self::Register {
        _mm256_movedup_pd(value)
    }
    #[inline(always)]
    unsafe fn dup_imag_components(value: Self::Register) -> Self::Register {
        _mm256_permute_pd(value, 0x0F)
    }
    #[inline(always)]
    unsafe fn swap_complex_components(value: Self::Register) -> Self::Register {
        _mm256_permute_pd(value, 0x5)
    }
    #[inline(always)]
    unsafe fn conj(value: Self::Register) -> Self::Register {
        _mm256_xor_pd(value, _mm256_setr_pd(-0.0, 0.0, -0.0, 0.0))
    }

    #[inline(always)]
    unsafe fn inv(value: Self::Register) -> Self::Register {
        let norm = <Avx2Complex as ComplexOps<f64>>::dup_norm(value);
        let conj = <Avx2Complex as ComplexOps<f64>>::conj(value);
        _mm256_div_pd(_mm256_div_pd(conj, norm), norm)
    }

    #[inline(always)]
    unsafe fn dup_norm(value: Self::Register) -> Self::Register {
        let (real, imag) =
            <Avx2Complex as ComplexOps<f64>>::dup_complex_components(value);

        _mm256_sqrt_pd(_mm256_add_pd(
            _mm256_mul_pd(real, real),
            _mm256_mul_pd(imag, imag),
        ))
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
        _mm256_setr_pd(value.im, value.re, value.im, value.re)
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
            <Avx2Complex as ComplexOps<f64>>::dup_complex_components(left);

        let right_shuffled =
            <Avx2Complex as ComplexOps<f64>>::swap_complex_components(right);

        let output_right = _mm256_mul_pd(left_imag, right_shuffled);
        let output_left = _mm256_mul_pd(left_real, right);

        _mm256_addsub_pd(output_left, output_right)
    }

    #[inline(always)]
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let l2_inv = <Avx2Complex as ComplexOps<f64>>::inv(l2);
        <Avx2Complex as SimdRegister<Complex<f64>>>::mul(l1, l2_inv)
    }

    #[inline(always)]
    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        <Avx2Complex as SimdRegister<Complex<f64>>>::add(
            <Avx2Complex as SimdRegister<Complex<f64>>>::mul(l1, l2),
            acc,
        )
    }
    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = <Avx2 as SimdRegister<f64>>::eq(l1, l2);
        _mm256_and_pd(mask, _mm256_permute_pd(mask, 0x5))
    }
    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = <Avx2 as SimdRegister<f64>>::neq(l1, l2);
        _mm256_andnot_pd(mask, _mm256_permute_pd(mask, 0x5))
    }
    #[inline(always)]
    unsafe fn write(mem: *mut Complex<f64>, reg: Self::Register) {
        _mm256_storeu_pd(mem as *mut f64, reg);
    }

    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> Complex<f64> {
        let (right_half, left_half) =
            (_mm256_castpd256_pd128(reg), _mm256_extractf128_pd::<1>(reg));
        let sum = _mm_add_pd(left_half, right_half);
        let mut store = [Complex::new(0.0, 0.0); 1];
        _mm_store_pd(store.as_mut_ptr() as *mut f64, sum);
        store[0]
    }

    unsafe fn max(_l1: Self::Register, _l22: Self::Register) -> Self::Register {
        unimplemented!("Ordering operations are not supported for complex numbers");
    }

    unsafe fn min(_l1: Self::Register, _l2: Self::Register) -> Self::Register {
        unimplemented!("Ordering operations are not supported for complex numbers");
    }

    unsafe fn lt(_l1: Self::Register, _l22: Self::Register) -> Self::Register {
        unimplemented!("Ordering operations are not supported for complex numbers");
    }

    unsafe fn lte(_l1: Self::Register, _l2: Self::Register) -> Self::Register {
        unimplemented!("Ordering operations are not supported for complex numbers");
    }

    unsafe fn gt(_l1: Self::Register, _l22: Self::Register) -> Self::Register {
        unimplemented!("Ordering operations are not supported for complex numbers");
    }

    unsafe fn gte(_l1: Self::Register, _l2: Self::Register) -> Self::Register {
        unimplemented!("Ordering operations are not supported for complex numbers");
    }

    unsafe fn max_to_value(_reg: Self::Register) -> Complex<f64> {
        unimplemented!("Ordering operations are not supported for complex numbers");
    }

    unsafe fn min_to_value(_reg: Self::Register) -> Complex<f64> {
        unimplemented!("Ordering operations are not supported for complex numbers");
    }
}
