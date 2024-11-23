#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::core_simd_api::{Hypot, SimdRegister};
use super::impl_avx2::Avx2;

/// AVX2 & FMA enabled SIMD operations.
///
/// This requires the `avx2` & `fma` CPU features be enabled.
pub struct Avx2Fma;

impl SimdRegister<f32> for Avx2Fma {
    type Register = __m256;

    #[inline(always)]
    unsafe fn load(mem: *const f32) -> Self::Register {
        Avx2::load(mem)
    }

    #[inline(always)]
    unsafe fn filled(value: f32) -> Self::Register {
        Avx2::filled(value)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        <Avx2 as SimdRegister<f32>>::zeroed()
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Avx2 as SimdRegister<f32>>::add(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Avx2 as SimdRegister<f32>>::sub(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Avx2 as SimdRegister<f32>>::mul(l1, l2)
    }

    #[inline(always)]
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Avx2 as SimdRegister<f32>>::div(l1, l2)
    }

    #[inline(always)]
    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        _mm256_fmadd_ps(l1, l2, acc)
    }

    #[inline(always)]
    unsafe fn max(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Avx2 as SimdRegister<f32>>::max(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Avx2 as SimdRegister<f32>>::min(l1, l2)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Avx2 as SimdRegister<f32>>::eq(l1, l2)
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Avx2 as SimdRegister<f32>>::neq(l1, l2)
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Avx2 as SimdRegister<f32>>::lt(l1, l2)
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Avx2 as SimdRegister<f32>>::lte(l1, l2)
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Avx2 as SimdRegister<f32>>::gt(l1, l2)
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Avx2 as SimdRegister<f32>>::gte(l1, l2)
    }

    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> f32 {
        Avx2::sum_to_value(reg)
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> f32 {
        Avx2::max_to_value(reg)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> f32 {
        Avx2::min_to_value(reg)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut f32, reg: Self::Register) {
        Avx2::write(mem, reg)
    }
}

impl Hypot<f32> for Avx2Fma {
    //b' * sqrt((a*b_hat)^2 + (b*b_hat)^2), where |b| > |a|, b'=(2^n<b), b_hat=1/b'
    #[inline(always)]
    unsafe fn hypot(x: Self::Register, y: Self::Register) -> Self::Register {
        // convert the inputs to absolute values
        let (x, y) = (
            _mm256_andnot_ps(_mm256_set1_ps(-0.0), x),
            _mm256_andnot_ps(_mm256_set1_ps(-0.0), y),
        );
        // find the max and min of the two inputs
        let (hi, lo) = (
            <Avx2 as SimdRegister<f32>>::max(x, y),
            <Avx2 as SimdRegister<f32>>::min(x, y),
        );
        //Infinity is represented by all 1s in the exponent, and all 0s in the mantissa
        let exponent_mask = _mm256_set1_ps(f32::INFINITY);
        //The mantissa mask is the inverse of the exponent mask
        let mantissa_mask = _mm256_sub_ps(
            _mm256_set1_ps(f32::MIN_POSITIVE),
            _mm256_set1_ps(f32::from_bits(1_u32)),
        );
        // round the hi values down to the nearest power of 2
        let hi2p = _mm256_and_ps(hi, exponent_mask);
        // we scale the values inside the root by the reciprocal of hi2p. since it's a power of 2,
        // we can double it and xor it with the exponent mask
        let scale = _mm256_xor_ps(_mm256_add_ps(hi2p, hi2p), exponent_mask);

        // create a mask that matches the normal hi values
        let mask = _mm256_cmp_ps::<_CMP_GT_OQ>(hi, _mm256_set1_ps(f32::MIN_POSITIVE));
        // replace the subnormal values of hi2p with the minimum positive normal value
        let outer_scale =
            _mm256_blendv_ps(_mm256_set1_ps(f32::MIN_POSITIVE), hi2p, mask);
        // replace the subnormal values of scale with the reciprocal of the minimum positive normal value
        let inner_scale =
            _mm256_blendv_ps(_mm256_set1_ps(1.0 / f32::MIN_POSITIVE), scale, mask);
        // create a mask that matches the subnormal hi values
        let mask = _mm256_cmp_ps::<_CMP_LT_OQ>(hi, _mm256_set1_ps(f32::MIN_POSITIVE));
        // since hi2p was preserved the exponent bits of hi, the exponent of hi/hi2p is 1
        let hi_scaled =
            _mm256_or_ps(_mm256_and_ps(hi, mantissa_mask), _mm256_set1_ps(1.0));
        // for the subnormal elements of hi, we need to subtract 1 from the scaled hi values
        let hi_scaled = _mm256_blendv_ps(
            hi_scaled,
            _mm256_sub_ps(hi_scaled, _mm256_set1_ps(1.0)),
            mask,
        );

        let hi_scaled = _mm256_mul_ps(hi_scaled, hi_scaled);
        let lo_scaled = _mm256_mul_ps(lo, inner_scale);
        _mm256_mul_ps(
            outer_scale,
            _mm256_sqrt_ps(_mm256_fmadd_ps(lo_scaled, lo_scaled, hi_scaled)),
        )
    }
}

impl SimdRegister<f64> for Avx2Fma {
    type Register = __m256d;

    #[inline(always)]
    unsafe fn load(mem: *const f64) -> Self::Register {
        Avx2::load(mem)
    }

    #[inline(always)]
    unsafe fn filled(value: f64) -> Self::Register {
        Avx2::filled(value)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        <Avx2 as SimdRegister<f64>>::zeroed()
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Avx2 as SimdRegister<f64>>::add(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Avx2 as SimdRegister<f64>>::sub(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Avx2 as SimdRegister<f64>>::mul(l1, l2)
    }

    #[inline(always)]
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Avx2 as SimdRegister<f64>>::div(l1, l2)
    }

    #[inline(always)]
    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        _mm256_fmadd_pd(l1, l2, acc)
    }

    #[inline(always)]
    unsafe fn max(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Avx2 as SimdRegister<f64>>::max(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Avx2 as SimdRegister<f64>>::min(l1, l2)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Avx2 as SimdRegister<f64>>::eq(l1, l2)
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Avx2 as SimdRegister<f64>>::neq(l1, l2)
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Avx2 as SimdRegister<f64>>::lt(l1, l2)
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Avx2 as SimdRegister<f64>>::lte(l1, l2)
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Avx2 as SimdRegister<f64>>::gt(l1, l2)
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Avx2 as SimdRegister<f64>>::gte(l1, l2)
    }
    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> f64 {
        Avx2::sum_to_value(reg)
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> f64 {
        Avx2::max_to_value(reg)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> f64 {
        Avx2::min_to_value(reg)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut f64, reg: Self::Register) {
        Avx2::write(mem, reg)
    }
}

impl Hypot<f64> for Avx2Fma {
    #[inline(always)]
    unsafe fn hypot(x: Self::Register, y: Self::Register) -> Self::Register {
        // convert the inputs to absolute values
        let (x, y) = (
            _mm256_andnot_pd(_mm256_set1_pd(-0.0), x),
            _mm256_andnot_pd(_mm256_set1_pd(-0.0), y),
        );
        // find the max and min of the two inputs
        let (hi, lo) = (
            <Avx2 as SimdRegister<f64>>::max(x, y),
            <Avx2 as SimdRegister<f64>>::min(x, y),
        );
        //Infinity is represented by all 1s in the exponent, and all 0s in the mantissa
        let exponent_mask = _mm256_set1_pd(f64::INFINITY);
        //The mantissa mask is the inverse of the exponent mask
        let mantissa_mask = _mm256_sub_pd(
            _mm256_set1_pd(f64::MIN_POSITIVE),
            _mm256_set1_pd(f64::from_bits(1_u64)),
        );
        // round the hi values down to the nearest power of 2
        let hi2p = _mm256_and_pd(hi, exponent_mask);
        // we scale the values inside the root by the reciprocal of hi2p. since it's a power of 2,
        // we can double it and xor it with the exponent mask
        let scale = _mm256_xor_pd(_mm256_add_pd(hi2p, hi2p), exponent_mask);

        // create a mask that matches the normal hi values
        let mask = _mm256_cmp_pd::<_CMP_GT_OQ>(hi, _mm256_set1_pd(f64::MIN_POSITIVE));
        // replace the subnormal values of hi2p with the minimum positive normal value
        let hi2p = _mm256_blendv_pd(_mm256_set1_pd(f64::MIN_POSITIVE), hi2p, mask);
        // replace the subnormal values of scale with the reciprocal of the minimum positive normal value
        let scale =
            _mm256_blendv_pd(_mm256_set1_pd(1.0 / f64::MIN_POSITIVE), scale, mask);
        // create a mask that matches the subnormal hi values
        let mask = _mm256_cmp_pd::<_CMP_LT_OQ>(hi, _mm256_set1_pd(f64::MIN_POSITIVE));
        // since hi2p was preserved the exponent bits of hi, the exponent of hi/hi2p is 1
        let hi_scaled =
            _mm256_or_pd(_mm256_and_pd(hi, mantissa_mask), _mm256_set1_pd(1.0));
        // for the subnormal elements of hi, we need to subtract 1 from the scaled hi values
        let hi_scaled = _mm256_blendv_pd(
            hi_scaled,
            _mm256_sub_pd(hi_scaled, _mm256_set1_pd(1.0)),
            mask,
        );

        let hi_scaled = _mm256_mul_pd(hi_scaled, hi_scaled);
        let lo_scaled = _mm256_mul_pd(lo, scale);

        _mm256_mul_pd(
            hi2p,
            _mm256_sqrt_pd(_mm256_fmadd_pd(lo_scaled, lo_scaled, hi_scaled)),
        )
    }
}
