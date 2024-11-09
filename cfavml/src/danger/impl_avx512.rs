#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::iter::zip;
use core::mem;

use super::core_simd_api::{DenseLane, Hypot, SimdRegister};
use super::impl_avx2::Avx2;
use crate::apply_dense;

/// AVX512 enabled SIMD operations.
///
/// This requires the `avx15f` CPU features be enabled.
///
/// NOTE:
///
/// Some impls may not be the most performant as Rust currently still has a limited
/// AVX512 support, in particular some utilities like AVX512DQ are currently unavailable.
pub struct Avx512;

impl SimdRegister<f32> for Avx512 {
    type Register = __m512;

    #[inline(always)]
    unsafe fn load(mem: *const f32) -> Self::Register {
        _mm512_loadu_ps(mem)
    }

    #[inline(always)]
    unsafe fn filled(value: f32) -> Self::Register {
        _mm512_set1_ps(value)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        _mm512_setzero_ps()
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_add_ps(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_sub_ps(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_mul_ps(l1, l2)
    }

    #[inline(always)]
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_div_ps(l1, l2)
    }

    #[inline(always)]
    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        _mm512_fmadd_ps(l1, l2, acc)
    }

    #[inline(always)]
    unsafe fn max(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_max_ps(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_min_ps(l1, l2)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(l1, l2);
        fast_cvt_mask16_to_m512(mask)
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmp_ps_mask::<_CMP_NEQ_UQ>(l1, l2);
        fast_cvt_mask16_to_m512(mask)
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(l1, l2);
        fast_cvt_mask16_to_m512(mask)
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmp_ps_mask::<_CMP_LE_OQ>(l1, l2);
        fast_cvt_mask16_to_m512(mask)
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(l1, l2);
        fast_cvt_mask16_to_m512(mask)
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(l1, l2);
        fast_cvt_mask16_to_m512(mask)
    }

    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> f32 {
        _mm512_reduce_add_ps(reg)
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> f32 {
        _mm512_reduce_max_ps(reg)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> f32 {
        _mm512_reduce_min_ps(reg)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut f32, reg: Self::Register) {
        _mm512_storeu_ps(mem, reg)
    }
}

impl Hypot<f32> for Avx512 {
    //b' * sqrt((a*b_hat)^2 + (b*b_hat)^2), where |b| > |a|, b'=(2^n<b), b_hat=1/b'
    #[inline(always)]
    unsafe fn hypot(x: Self::Register, y: Self::Register) -> Self::Register {
        // convert the inputs to absolute values
        let (x, y) = (
            _mm512_andnot_ps(_mm512_set1_ps(-0.0), x),
            _mm512_andnot_ps(_mm512_set1_ps(-0.0), y),
        );
        // find the max and min of the two inputs
        let (hi, lo) = (
            <Avx512 as SimdRegister<f32>>::max(x, y),
            <Avx512 as SimdRegister<f32>>::min(x, y),
        );
        //Infinity is represented by all 1s in the exponent, and all 0s in the mantissa
        let exponent_mask = _mm512_set1_ps(f32::INFINITY);
        //The mantissa mask is the inverse of the exponent mask
        let mantissa_mask = _mm512_sub_ps(
            _mm512_set1_ps(f32::MIN_POSITIVE),
            _mm512_set1_ps(core::mem::transmute(1_u32)),
        );
        // round the hi values down to the nearest power of 2
        let hi2p = _mm512_and_ps(hi, exponent_mask);
        // we scale the values inside the root by the reciprocal of hi2p. since it's a power of 2,
        // we can double it and xor it with the exponent mask
        let scale = _mm512_xor_ps(_mm512_add_ps(hi2p, hi2p), exponent_mask);

        // create a mask that matches the normal hi values
        let mask =
            _mm512_cmp_ps_mask::<_CMP_GT_OQ>(hi, _mm512_set1_ps(f32::MIN_POSITIVE));
        // replace the subnormal values of hi2p with the minimum positive normal value
        let hi2p = _mm512_mask_blend_ps(mask, _mm512_set1_ps(f32::MIN_POSITIVE), hi2p);
        // replace the subnormal values of scale with the reciprocal of the minimum positive normal value
        let scale =
            _mm512_mask_blend_ps(mask, _mm512_set1_ps(1.0 / f32::MIN_POSITIVE), scale);
        // create a mask that matches the subnormal hi values
        let mask =
            _mm512_cmp_ps_mask::<_CMP_LT_OQ>(hi, _mm512_set1_ps(f32::MIN_POSITIVE));
        // since hi2p was preserved the exponent bits of hi, the exponent of hi/hi2p is 1
        let hi_scaled =
            _mm512_or_ps(_mm512_and_ps(hi, mantissa_mask), _mm512_set1_ps(1.0));
        // for the subnormal elements of hi, we need to subtract 1 from the scaled hi values
        let hi_scaled = _mm512_mask_blend_ps(
            mask,
            hi_scaled,
            _mm512_sub_ps(hi_scaled, _mm512_set1_ps(1.0)),
        );

        let hi_scaled = _mm512_mul_ps(hi_scaled, hi_scaled);
        let lo_scaled = _mm512_mul_ps(lo, scale);
        _mm512_mul_ps(
            hi2p,
            _mm512_sqrt_ps(_mm512_fmadd_ps(lo_scaled, lo_scaled, hi_scaled)),
        )
    }
}

impl SimdRegister<f64> for Avx512 {
    type Register = __m512d;

    #[inline(always)]
    unsafe fn load(mem: *const f64) -> Self::Register {
        _mm512_loadu_pd(mem)
    }

    #[inline(always)]
    unsafe fn filled(value: f64) -> Self::Register {
        _mm512_set1_pd(value)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        _mm512_setzero_pd()
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_add_pd(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_sub_pd(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_mul_pd(l1, l2)
    }

    #[inline(always)]
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_div_pd(l1, l2)
    }

    #[inline(always)]
    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        _mm512_fmadd_pd(l1, l2, acc)
    }

    #[inline(always)]
    unsafe fn max(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_max_pd(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_min_pd(l1, l2)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmp_pd_mask::<_CMP_EQ_OQ>(l1, l2);
        fast_cvt_mask8_to_m512d(mask)
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmp_pd_mask::<_CMP_NEQ_UQ>(l1, l2);
        fast_cvt_mask8_to_m512d(mask)
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmp_pd_mask::<_CMP_LT_OQ>(l1, l2);
        fast_cvt_mask8_to_m512d(mask)
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(l1, l2);
        fast_cvt_mask8_to_m512d(mask)
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmp_pd_mask::<_CMP_GT_OQ>(l1, l2);
        fast_cvt_mask8_to_m512d(mask)
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(l1, l2);
        fast_cvt_mask8_to_m512d(mask)
    }

    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> f64 {
        _mm512_reduce_add_pd(reg)
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> f64 {
        _mm512_reduce_max_pd(reg)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> f64 {
        _mm512_reduce_min_pd(reg)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut f64, reg: Self::Register) {
        _mm512_storeu_pd(mem, reg)
    }
}

impl Hypot<f64> for Avx512 {
    #[inline(always)]
    unsafe fn hypot(x: Self::Register, y: Self::Register) -> Self::Register {
        // convert the inputs to absolute values
        let (x, y) = (
            _mm512_andnot_pd(_mm512_set1_pd(-0.0), x),
            _mm512_andnot_pd(_mm512_set1_pd(-0.0), y),
        );
        // find the max and min of the two inputs
        let (hi, lo) = (
            <Avx512 as SimdRegister<f64>>::max(x, y),
            <Avx512 as SimdRegister<f64>>::min(x, y),
        );
        //Infinity is represented by all 1s in the exponent, and all 0s in the mantissa
        let exponent_mask = _mm512_set1_pd(f64::INFINITY);
        //The mantissa mask is the inverse of the exponent mask
        let mantissa_mask = _mm512_sub_pd(
            _mm512_set1_pd(f64::MIN_POSITIVE),
            _mm512_set1_pd(core::mem::transmute(1_u64)),
        );
        // round the hi values down to the nearest power of 2
        let hi2p = _mm512_and_pd(hi, exponent_mask);
        // we scale the values inside the root by the reciprocal of hi2p. since it's a power of 2,
        // we can double it and xor it with the exponent mask
        let scale = _mm512_xor_pd(_mm512_add_pd(hi2p, hi2p), exponent_mask);

        // create a mask that matches the normal hi values
        let mask =
            _mm512_cmp_pd_mask::<_CMP_GT_OQ>(hi, _mm512_set1_pd(f64::MIN_POSITIVE));
        // replace the subnormal values of hi2p with the minimum positive normal value
        let hi2p = _mm512_mask_blend_pd(mask, _mm512_set1_pd(f64::MIN_POSITIVE), hi2p);
        // replace the subnormal values of scale with the reciprocal of the minimum positive normal value
        let scale =
            _mm512_mask_blend_pd(mask, _mm512_set1_pd(1.0 / f64::MIN_POSITIVE), scale);
        // create a mask that matches the subnormal hi values
        let mask =
            _mm512_cmp_pd_mask::<_CMP_LT_OQ>(hi, _mm512_set1_pd(f64::MIN_POSITIVE));
        // since hi2p was preserved the exponent bits of hi, the exponent of hi/hi2p is 1
        let hi_scaled =
            _mm512_or_pd(_mm512_and_pd(hi, mantissa_mask), _mm512_set1_pd(1.0));
        // for the subnormal elements of hi, we need to subtract 1 from the scaled hi values
        let hi_scaled = _mm512_mask_blend_pd(
            mask,
            hi_scaled,
            _mm512_sub_pd(hi_scaled, _mm512_set1_pd(1.0)),
        );

        let hi_scaled = _mm512_mul_pd(hi_scaled, hi_scaled);
        let lo_scaled = _mm512_mul_pd(lo, scale);

        _mm512_mul_pd(
            hi2p,
            _mm512_sqrt_pd(_mm512_fmadd_pd(lo_scaled, lo_scaled, hi_scaled)),
        )
    }
}
impl SimdRegister<i8> for Avx512 {
    type Register = __m512i;

    #[inline(always)]
    unsafe fn load(mem: *const i8) -> Self::Register {
        _mm512_loadu_si512(mem.cast())
    }

    #[inline(always)]
    unsafe fn filled(value: i8) -> Self::Register {
        _mm512_set1_epi8(value)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        _mm512_setzero_si512()
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_add_epi8(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_sub_epi8(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let shift_l1 = _mm512_srai_epi16::<8>(l1);
        let shift_l2 = _mm512_srai_epi16::<8>(l2);

        let even = _mm512_mullo_epi16(l1, l2);
        let odd = _mm512_mullo_epi16(shift_l1, shift_l2);
        let odd = _mm512_slli_epi16::<8>(odd);
        _mm512_mask_blend_epi8(0xAAAAAAAAAAAAAAAA, even, odd)
    }

    #[inline(always)]
    /// Scalar `i8` integer division.
    ///
    /// In reality this operation is not SIMD, in theory we could later support
    /// it however it will always be an incredibly expensive operation with quite
    /// a lot of cognitive load on the maintenance side, so for the foreseeable future
    /// non-floating point division operations will be non-simd.
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let l1_unpacked = mem::transmute::<_, [i8; 64]>(l1);
        let l2_unpacked = mem::transmute::<_, [i8; 64]>(l2);

        let mut result = [0i8; 64];
        for (idx, (l1, l2)) in zip(l1_unpacked, l2_unpacked).enumerate() {
            result[idx] = l1.wrapping_div(l2);
        }

        mem::transmute::<_, Self::Register>(result)
    }

    #[inline(always)]
    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        let res = <Self as SimdRegister<i8>>::mul(l1, l2);
        <Self as SimdRegister<i8>>::add(res, acc)
    }

    #[inline(always)]
    unsafe fn max(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_max_epi8(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_min_epi8(l1, l2)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpeq_epi8_mask(l1, l2);
        fast_cvt_mask64_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpneq_epi8_mask(l1, l2);
        fast_cvt_mask64_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmplt_epi8_mask(l1, l2);
        fast_cvt_mask64_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmple_epi8_mask(l1, l2);
        fast_cvt_mask64_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpgt_epi8_mask(l1, l2);
        fast_cvt_mask64_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpge_epi8_mask(l1, l2);
        fast_cvt_mask64_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn mul_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let mask = DenseLane::copy(0xAAAAAAAAAAAAAAAA);

        let shift_l1 = apply_dense!(_mm512_srai_epi16::<8>, l1);
        let shift_l2 = apply_dense!(_mm512_srai_epi16::<8>, l2);

        let even = apply_dense!(_mm512_mullo_epi16, l1, l2);
        let odd = apply_dense!(_mm512_mullo_epi16, shift_l1, shift_l2);
        let odd = apply_dense!(_mm512_slli_epi16::<8>, odd);

        apply_dense!(_mm512_mask_blend_epi8, mask, even, odd)
    }

    #[inline(always)]
    unsafe fn fmadd_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
        acc: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let res = <Self as SimdRegister<i8>>::mul_dense(l1, l2);
        <Self as SimdRegister<i8>>::add_dense(res, acc)
    }

    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> i8 {
        // TODO: This is probably not the most optimal way of doing this.
        let swapped =
            _mm512_shuffle_i64x2::<{ super::_MM_SHUFFLE(1, 0, 3, 2) }>(reg, reg);
        let hi = _mm512_castsi512_si256(swapped);
        let lo = _mm512_castsi512_si256(reg);
        let sum = <Avx2 as SimdRegister<i8>>::add(hi, lo);
        <Avx2 as SimdRegister<i8>>::sum_to_value(sum)
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> i8 {
        let swapped =
            _mm512_shuffle_i64x2::<{ super::_MM_SHUFFLE(1, 0, 3, 2) }>(reg, reg);
        let hi = _mm512_castsi512_si256(swapped);
        let lo = _mm512_castsi512_si256(reg);
        let max = <Avx2 as SimdRegister<i8>>::max(hi, lo);
        <Avx2 as SimdRegister<i8>>::max_to_value(max)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> i8 {
        let swapped =
            _mm512_shuffle_i64x2::<{ super::_MM_SHUFFLE(1, 0, 3, 2) }>(reg, reg);
        let hi = _mm512_castsi512_si256(swapped);
        let lo = _mm512_castsi512_si256(reg);
        let max = <Avx2 as SimdRegister<i8>>::min(hi, lo);
        <Avx2 as SimdRegister<i8>>::min_to_value(max)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut i8, reg: Self::Register) {
        _mm512_storeu_si512(mem.cast(), reg)
    }
}

impl SimdRegister<i16> for Avx512 {
    type Register = __m512i;

    #[inline(always)]
    unsafe fn load(mem: *const i16) -> Self::Register {
        _mm512_loadu_si512(mem.cast())
    }

    #[inline(always)]
    unsafe fn filled(value: i16) -> Self::Register {
        _mm512_set1_epi16(value)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        _mm512_setzero_si512()
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_add_epi16(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_sub_epi16(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_mullo_epi16(l1, l2)
    }

    #[inline(always)]
    /// Scalar `i16` integer division.
    ///
    /// In reality this operation is not SIMD, in theory we could later support
    /// it however it will always be an incredibly expensive operation with quite
    /// a lot of cognitive load on the maintenance side, so for the foreseeable future
    /// non-floating point division operations will be non-simd.
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let l1_unpacked = mem::transmute::<_, [i16; 32]>(l1);
        let l2_unpacked = mem::transmute::<_, [i16; 32]>(l2);

        let mut result = [0i16; 32];
        for (idx, (l1, l2)) in zip(l1_unpacked, l2_unpacked).enumerate() {
            result[idx] = l1.wrapping_div(l2);
        }

        mem::transmute::<_, Self::Register>(result)
    }

    #[inline(always)]
    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        // A non-fused variant for AVX2 without the FMA cpu feature requirement.
        let res = <Self as SimdRegister<i16>>::mul(l1, l2);
        <Self as SimdRegister<i16>>::add(res, acc)
    }

    #[inline(always)]
    unsafe fn max(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_max_epi16(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_min_epi16(l1, l2)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpeq_epi16_mask(l1, l2);
        fast_cvt_mask32_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpneq_epi16_mask(l1, l2);
        fast_cvt_mask32_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmplt_epi16_mask(l1, l2);
        fast_cvt_mask32_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmple_epi16_mask(l1, l2);
        fast_cvt_mask32_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpgt_epi16_mask(l1, l2);
        fast_cvt_mask32_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpge_epi16_mask(l1, l2);
        fast_cvt_mask32_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn mul_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        apply_dense!(_mm512_mullo_epi16, l1, l2)
    }

    #[inline(always)]
    unsafe fn fmadd_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
        acc: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let res = <Self as SimdRegister<i16>>::mul_dense(l1, l2);
        <Self as SimdRegister<i16>>::add_dense(res, acc)
    }

    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> i16 {
        let swapped =
            _mm512_shuffle_i64x2::<{ super::_MM_SHUFFLE(1, 0, 3, 2) }>(reg, reg);
        let hi = _mm512_castsi512_si256(swapped);
        let lo = _mm512_castsi512_si256(reg);
        let max = <Avx2 as SimdRegister<i16>>::add(hi, lo);
        <Avx2 as SimdRegister<i16>>::sum_to_value(max)
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> i16 {
        let swapped =
            _mm512_shuffle_i64x2::<{ super::_MM_SHUFFLE(1, 0, 3, 2) }>(reg, reg);
        let hi = _mm512_castsi512_si256(swapped);
        let lo = _mm512_castsi512_si256(reg);
        let max = <Avx2 as SimdRegister<i16>>::max(hi, lo);
        <Avx2 as SimdRegister<i16>>::max_to_value(max)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> i16 {
        let swapped =
            _mm512_shuffle_i64x2::<{ super::_MM_SHUFFLE(1, 0, 3, 2) }>(reg, reg);
        let hi = _mm512_castsi512_si256(swapped);
        let lo = _mm512_castsi512_si256(reg);
        let max = <Avx2 as SimdRegister<i16>>::min(hi, lo);
        <Avx2 as SimdRegister<i16>>::min_to_value(max)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut i16, reg: Self::Register) {
        _mm512_storeu_si512(mem.cast(), reg)
    }
}

impl SimdRegister<i32> for Avx512 {
    type Register = __m512i;

    #[inline(always)]
    unsafe fn load(mem: *const i32) -> Self::Register {
        _mm512_loadu_si512(mem.cast())
    }

    #[inline(always)]
    unsafe fn filled(value: i32) -> Self::Register {
        _mm512_set1_epi32(value)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        _mm512_setzero_si512()
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_add_epi32(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_sub_epi32(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_mullo_epi32(l1, l2)
    }

    #[inline(always)]
    /// Scalar `i32` integer division.
    ///
    /// In reality this operation is not SIMD, in theory we could later support
    /// it however it will always be an incredibly expensive operation with quite
    /// a lot of cognitive load on the maintenance side, so for the foreseeable future
    /// non-floating point division operations will be non-simd.
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let l1_unpacked = mem::transmute::<_, [i32; 16]>(l1);
        let l2_unpacked = mem::transmute::<_, [i32; 16]>(l2);

        let mut result = [0i32; 16];
        for (idx, (l1, l2)) in zip(l1_unpacked, l2_unpacked).enumerate() {
            result[idx] = l1.wrapping_div(l2);
        }

        mem::transmute::<_, Self::Register>(result)
    }

    #[inline(always)]
    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        // A non-fused variant for AVX2 without the FMA cpu feature requirement.
        let res = <Self as SimdRegister<i32>>::mul(l1, l2);
        <Self as SimdRegister<i32>>::add(res, acc)
    }

    #[inline(always)]
    unsafe fn max(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_max_epi32(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_min_epi32(l1, l2)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpeq_epi32_mask(l1, l2);
        fast_cvt_mask16_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpneq_epi32_mask(l1, l2);
        fast_cvt_mask16_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmplt_epi32_mask(l1, l2);
        fast_cvt_mask16_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmple_epi32_mask(l1, l2);
        fast_cvt_mask16_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpgt_epi32_mask(l1, l2);
        fast_cvt_mask16_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpge_epi32_mask(l1, l2);
        fast_cvt_mask16_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn mul_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        apply_dense!(_mm512_mullo_epi32, l1, l2)
    }

    #[inline(always)]
    unsafe fn fmadd_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
        acc: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let res = <Self as SimdRegister<i32>>::mul_dense(l1, l2);
        <Self as SimdRegister<i32>>::add_dense(res, acc)
    }

    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> i32 {
        _mm512_reduce_add_epi32(reg)
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> i32 {
        _mm512_reduce_max_epi32(reg)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> i32 {
        _mm512_reduce_min_epi32(reg)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut i32, reg: Self::Register) {
        _mm512_storeu_si512(mem.cast(), reg)
    }
}

impl SimdRegister<i64> for Avx512 {
    type Register = __m512i;

    #[inline(always)]
    unsafe fn load(mem: *const i64) -> Self::Register {
        _mm512_loadu_si512(mem.cast())
    }

    #[inline(always)]
    unsafe fn filled(value: i64) -> Self::Register {
        _mm512_set1_epi64(value)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        _mm512_setzero_si512()
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_add_epi64(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_sub_epi64(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        // TODO: Evaluate VS doing what we did on AVX2...
        _mm512_mullox_epi64(l1, l2)
    }

    #[inline(always)]
    /// Scalar `i64` integer division.
    ///
    /// In reality this operation is not SIMD, in theory we could later support
    /// it however it will always be an incredibly expensive operation with quite
    /// a lot of cognitive load on the maintenance side, so for the foreseeable future
    /// non-floating point division operations will be non-simd.
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let l1_unpacked = mem::transmute::<_, [i64; 8]>(l1);
        let l2_unpacked = mem::transmute::<_, [i64; 8]>(l2);

        let mut result = [0i64; 8];
        for (idx, (l1, l2)) in zip(l1_unpacked, l2_unpacked).enumerate() {
            result[idx] = l1.wrapping_div(l2);
        }

        mem::transmute::<_, Self::Register>(result)
    }

    #[inline(always)]
    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        // A non-fused variant for AVX2 without the FMA cpu feature requirement.
        let res = <Self as SimdRegister<i64>>::mul(l1, l2);
        <Self as SimdRegister<i64>>::add(res, acc)
    }

    #[inline(always)]
    unsafe fn max(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_max_epi64(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_min_epi64(l1, l2)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpeq_epi64_mask(l1, l2);
        fast_cvt_mask8_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpneq_epi64_mask(l1, l2);
        fast_cvt_mask8_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmplt_epi64_mask(l1, l2);
        fast_cvt_mask8_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmple_epi64_mask(l1, l2);
        fast_cvt_mask8_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpgt_epi64_mask(l1, l2);
        fast_cvt_mask8_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpge_epi64_mask(l1, l2);
        fast_cvt_mask8_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn fmadd_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
        acc: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let res = <Self as SimdRegister<i64>>::mul_dense(l1, l2);
        <Self as SimdRegister<i64>>::add_dense(res, acc)
    }

    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> i64 {
        _mm512_reduce_add_epi64(reg)
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> i64 {
        _mm512_reduce_max_epi64(reg)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> i64 {
        _mm512_reduce_min_epi64(reg)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut i64, reg: Self::Register) {
        _mm512_storeu_si512(mem.cast(), reg)
    }
}

impl SimdRegister<u8> for Avx512 {
    type Register = __m512i;

    #[inline(always)]
    unsafe fn load(mem: *const u8) -> Self::Register {
        _mm512_loadu_si512(mem.cast())
    }

    #[inline(always)]
    unsafe fn filled(value: u8) -> Self::Register {
        _mm512_set1_epi8(value as i8)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        _mm512_setzero_si512()
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_add_epi8(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_sub_epi8(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Self as SimdRegister<i8>>::mul(l1, l2)
    }

    #[inline(always)]
    /// Scalar `u8` integer division.
    ///
    /// In reality this operation is not SIMD, in theory we could later support
    /// it however it will always be an incredibly expensive operation with quite
    /// a lot of cognitive load on the maintenance side, so for the foreseeable future
    /// non-floating point division operations will be non-simd.
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let l1_unpacked = mem::transmute::<_, [u8; 64]>(l1);
        let l2_unpacked = mem::transmute::<_, [u8; 64]>(l2);

        let mut result = [0u8; 64];
        for (idx, (l1, l2)) in zip(l1_unpacked, l2_unpacked).enumerate() {
            result[idx] = l1.wrapping_div(l2);
        }

        mem::transmute::<_, Self::Register>(result)
    }

    #[inline(always)]
    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        let res = <Self as SimdRegister<u8>>::mul(l1, l2);
        <Self as SimdRegister<u8>>::add(res, acc)
    }

    #[inline(always)]
    unsafe fn max(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_max_epu8(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_min_epu8(l1, l2)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpeq_epu8_mask(l1, l2);
        fast_cvt_mask64_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpneq_epu8_mask(l1, l2);
        fast_cvt_mask64_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmplt_epu8_mask(l1, l2);
        fast_cvt_mask64_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmple_epu8_mask(l1, l2);
        fast_cvt_mask64_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpgt_epu8_mask(l1, l2);
        fast_cvt_mask64_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpge_epu8_mask(l1, l2);
        fast_cvt_mask64_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn mul_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        <Self as SimdRegister<i8>>::mul_dense(l1, l2)
    }

    #[inline(always)]
    unsafe fn fmadd_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
        acc: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let res = <Self as SimdRegister<u8>>::mul_dense(l1, l2);
        <Self as SimdRegister<u8>>::add_dense(res, acc)
    }

    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> u8 {
        <Self as SimdRegister<i8>>::sum_to_value(reg) as u8
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> u8 {
        let swapped =
            _mm512_shuffle_i64x2::<{ super::_MM_SHUFFLE(1, 0, 3, 2) }>(reg, reg);
        let hi = _mm512_castsi512_si256(swapped);
        let lo = _mm512_castsi512_si256(reg);
        let max = <Avx2 as SimdRegister<u8>>::max(hi, lo);
        <Avx2 as SimdRegister<u8>>::max_to_value(max)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> u8 {
        let swapped =
            _mm512_shuffle_i64x2::<{ super::_MM_SHUFFLE(1, 0, 3, 2) }>(reg, reg);
        let hi = _mm512_castsi512_si256(swapped);
        let lo = _mm512_castsi512_si256(reg);
        let max = <Avx2 as SimdRegister<u8>>::min(hi, lo);
        <Avx2 as SimdRegister<u8>>::min_to_value(max)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut u8, reg: Self::Register) {
        _mm512_storeu_si512(mem.cast(), reg)
    }
}

impl SimdRegister<u16> for Avx512 {
    type Register = __m512i;

    #[inline(always)]
    unsafe fn load(mem: *const u16) -> Self::Register {
        _mm512_loadu_si512(mem.cast())
    }

    #[inline(always)]
    unsafe fn filled(value: u16) -> Self::Register {
        _mm512_set1_epi16(value as i16)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        _mm512_setzero_si512()
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_add_epi16(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_sub_epi16(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_mullo_epi16(l1, l2)
    }

    #[inline(always)]
    /// Scalar `u16` integer division.
    ///
    /// In reality this operation is not SIMD, in theory we could later support
    /// it however it will always be an incredibly expensive operation with quite
    /// a lot of cognitive load on the maintenance side, so for the foreseeable future
    /// non-floating point division operations will be non-simd.
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let l1_unpacked = mem::transmute::<_, [u16; 32]>(l1);
        let l2_unpacked = mem::transmute::<_, [u16; 32]>(l2);

        let mut result = [0u16; 32];
        for (idx, (l1, l2)) in zip(l1_unpacked, l2_unpacked).enumerate() {
            result[idx] = l1.wrapping_div(l2);
        }

        mem::transmute::<_, Self::Register>(result)
    }

    #[inline(always)]
    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        let res = <Self as SimdRegister<u16>>::mul(l1, l2);
        <Self as SimdRegister<u16>>::add(res, acc)
    }

    #[inline(always)]
    unsafe fn max(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_max_epu16(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_min_epu16(l1, l2)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpeq_epu16_mask(l1, l2);
        fast_cvt_mask32_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpneq_epu16_mask(l1, l2);
        fast_cvt_mask32_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmplt_epu16_mask(l1, l2);
        fast_cvt_mask32_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmple_epu16_mask(l1, l2);
        fast_cvt_mask32_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpgt_epu16_mask(l1, l2);
        fast_cvt_mask32_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpge_epu16_mask(l1, l2);
        fast_cvt_mask32_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn mul_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        <Self as SimdRegister<i16>>::mul_dense(l1, l2)
    }

    #[inline(always)]
    unsafe fn fmadd_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
        acc: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let res = <Self as SimdRegister<u16>>::mul_dense(l1, l2);
        <Self as SimdRegister<u16>>::add_dense(res, acc)
    }

    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> u16 {
        <Self as SimdRegister<i16>>::sum_to_value(reg) as u16
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> u16 {
        let swapped =
            _mm512_shuffle_i64x2::<{ super::_MM_SHUFFLE(1, 0, 3, 2) }>(reg, reg);
        let hi = _mm512_castsi512_si256(swapped);
        let lo = _mm512_castsi512_si256(reg);
        let max = <Avx2 as SimdRegister<u16>>::max(hi, lo);
        <Avx2 as SimdRegister<u16>>::max_to_value(max)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> u16 {
        let swapped =
            _mm512_shuffle_i64x2::<{ super::_MM_SHUFFLE(1, 0, 3, 2) }>(reg, reg);
        let hi = _mm512_castsi512_si256(swapped);
        let lo = _mm512_castsi512_si256(reg);
        let max = <Avx2 as SimdRegister<u16>>::min(hi, lo);
        <Avx2 as SimdRegister<u16>>::min_to_value(max)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut u16, reg: Self::Register) {
        _mm512_storeu_si512(mem.cast(), reg)
    }
}

impl SimdRegister<u32> for Avx512 {
    type Register = __m512i;

    #[inline(always)]
    unsafe fn load(mem: *const u32) -> Self::Register {
        _mm512_loadu_si512(mem.cast())
    }

    #[inline(always)]
    unsafe fn filled(value: u32) -> Self::Register {
        _mm512_set1_epi32(value as i32)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        _mm512_setzero_si512()
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_add_epi32(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_sub_epi32(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_mullo_epi32(l1, l2)
    }

    #[inline(always)]
    /// Scalar `u32` integer division.
    ///
    /// In reality this operation is not SIMD, in theory we could later support
    /// it however it will always be an incredibly expensive operation with quite
    /// a lot of cognitive load on the maintenance side, so for the foreseeable future
    /// non-floating point division operations will be non-simd.
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let l1_unpacked = mem::transmute::<_, [u32; 16]>(l1);
        let l2_unpacked = mem::transmute::<_, [u32; 16]>(l2);

        let mut result = [0u32; 16];
        for (idx, (l1, l2)) in zip(l1_unpacked, l2_unpacked).enumerate() {
            result[idx] = l1.wrapping_div(l2);
        }

        mem::transmute::<_, Self::Register>(result)
    }

    #[inline(always)]
    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        let res = <Self as SimdRegister<u32>>::mul(l1, l2);
        <Self as SimdRegister<u32>>::add(res, acc)
    }

    #[inline(always)]
    unsafe fn max(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_max_epu32(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_min_epu32(l1, l2)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpeq_epu32_mask(l1, l2);
        fast_cvt_mask16_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpneq_epu32_mask(l1, l2);
        fast_cvt_mask16_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmplt_epu32_mask(l1, l2);
        fast_cvt_mask16_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmple_epu32_mask(l1, l2);
        fast_cvt_mask16_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpgt_epu32_mask(l1, l2);
        fast_cvt_mask16_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpge_epu32_mask(l1, l2);
        fast_cvt_mask16_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn mul_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        <Self as SimdRegister<i32>>::mul_dense(l1, l2)
    }

    #[inline(always)]
    unsafe fn fmadd_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
        acc: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let res = <Self as SimdRegister<u32>>::mul_dense(l1, l2);
        <Self as SimdRegister<u32>>::add_dense(res, acc)
    }

    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> u32 {
        _mm512_reduce_add_epi32(reg) as u32
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> u32 {
        _mm512_reduce_max_epu32(reg)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> u32 {
        _mm512_reduce_min_epu32(reg)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut u32, reg: Self::Register) {
        _mm512_storeu_si512(mem.cast(), reg)
    }
}

impl SimdRegister<u64> for Avx512 {
    type Register = __m512i;

    #[inline(always)]
    unsafe fn load(mem: *const u64) -> Self::Register {
        _mm512_loadu_si512(mem.cast())
    }

    #[inline(always)]
    unsafe fn filled(value: u64) -> Self::Register {
        _mm512_set1_epi64(value as i64)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        _mm512_setzero_si512()
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_add_epi64(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_sub_epi64(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Self as SimdRegister<i64>>::mul(l1, l2)
    }

    #[inline(always)]
    /// Scalar `u64` integer division.
    ///
    /// In reality this operation is not SIMD, in theory we could later support
    /// it however it will always be an incredibly expensive operation with quite
    /// a lot of cognitive load on the maintenance side, so for the foreseeable future
    /// non-floating point division operations will be non-simd.
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let l1_unpacked = mem::transmute::<_, [u64; 8]>(l1);
        let l2_unpacked = mem::transmute::<_, [u64; 8]>(l2);

        let mut result = [0u64; 8];
        for (idx, (l1, l2)) in zip(l1_unpacked, l2_unpacked).enumerate() {
            result[idx] = l1.wrapping_div(l2);
        }

        mem::transmute::<_, Self::Register>(result)
    }

    #[inline(always)]
    unsafe fn mul_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        <Self as SimdRegister<i64>>::mul_dense(l1, l2)
    }

    #[inline(always)]
    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        // A non-fused variant for AVX2 without the FMA cpu feature requirement.
        let res = <Self as SimdRegister<u64>>::mul(l1, l2);
        <Self as SimdRegister<u64>>::add(res, acc)
    }

    #[inline(always)]
    unsafe fn max(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_max_epu64(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm512_min_epu64(l1, l2)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpeq_epu64_mask(l1, l2);
        fast_cvt_mask8_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpneq_epu64_mask(l1, l2);
        fast_cvt_mask8_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmplt_epu64_mask(l1, l2);
        fast_cvt_mask8_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmple_epu64_mask(l1, l2);
        fast_cvt_mask8_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpgt_epu64_mask(l1, l2);
        fast_cvt_mask8_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm512_cmpge_epu64_mask(l1, l2);
        fast_cvt_mask8_to_m512i(mask)
    }

    #[inline(always)]
    unsafe fn fmadd_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
        acc: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let res = <Self as SimdRegister<u64>>::mul_dense(l1, l2);
        <Self as SimdRegister<u64>>::add_dense(res, acc)
    }

    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> u64 {
        _mm512_reduce_add_epi64(reg) as u64
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> u64 {
        _mm512_reduce_max_epu64(reg)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> u64 {
        _mm512_reduce_min_epu64(reg)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut u64, reg: Self::Register) {
        _mm512_storeu_si512(mem.cast(), reg)
    }
}

#[inline(always)]
unsafe fn fast_cvt_mask64_to_m512i(mask: __mmask64) -> __m512i {
    let zeroes = _mm512_setzero_si512();
    let ones = _mm512_set1_epi8(1);
    _mm512_mask_sub_epi8(zeroes, mask, ones, zeroes)
}

#[inline(always)]
unsafe fn fast_cvt_mask32_to_m512i(mask: __mmask32) -> __m512i {
    let zeroes = _mm512_setzero_si512();
    let ones = _mm512_set1_epi16(1);
    _mm512_mask_sub_epi16(zeroes, mask, ones, zeroes)
}

#[inline(always)]
unsafe fn fast_cvt_mask16_to_m512i(mask: __mmask16) -> __m512i {
    let zeroes = _mm512_setzero_si512();
    let ones = _mm512_set1_epi32(1);
    _mm512_mask_sub_epi32(zeroes, mask, ones, zeroes)
}

#[inline(always)]
unsafe fn fast_cvt_mask8_to_m512i(mask: __mmask8) -> __m512i {
    let zeroes = _mm512_setzero_si512();
    let ones = _mm512_set1_epi64(1);
    _mm512_mask_sub_epi64(zeroes, mask, ones, zeroes)
}

#[inline(always)]
unsafe fn fast_cvt_mask16_to_m512(mask: __mmask16) -> __m512 {
    let zeroes = _mm512_setzero_si512();
    let ones = _mm512_set1_ps(1.0);
    let expanded_mask =
        _mm512_mask_sub_epi32(zeroes, mask, _mm512_castps_si512(ones), zeroes);
    _mm512_castsi512_ps(expanded_mask)
}

#[inline(always)]
unsafe fn fast_cvt_mask8_to_m512d(mask: __mmask8) -> __m512d {
    let zeroes = _mm512_setzero_si512();
    let ones = _mm512_set1_pd(1.0);
    let expanded_mask =
        _mm512_mask_sub_epi64(zeroes, mask, _mm512_castpd_si512(ones), zeroes);
    _mm512_castsi512_pd(expanded_mask)
}
