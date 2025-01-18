#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::iter::zip;
use core::mem;

use super::core_simd_api::{DenseLane, Hypot, SimdRegister};
use crate::apply_dense;

/// AVX2 enabled SIMD operations.
///
/// This requires the `avx2` CPU features be enabled, but does not required FMA.
pub struct Avx2;

impl SimdRegister<f32> for Avx2 {
    type Register = __m256;

    #[inline(always)]
    unsafe fn load(mem: *const f32) -> Self::Register {
        _mm256_loadu_ps(mem)
    }

    #[inline(always)]
    unsafe fn filled(value: f32) -> Self::Register {
        _mm256_set1_ps(value)
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
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_mul_ps(l1, l2)
    }

    #[inline(always)]
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_div_ps(l1, l2)
    }

    #[inline(always)]
    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        let res = <Self as SimdRegister<f32>>::mul(l1, l2);
        <Self as SimdRegister<f32>>::add(res, acc)
    }

    #[inline(always)]
    unsafe fn max(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_max_ps(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_min_ps(l1, l2)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm256_cmp_ps::<_CMP_EQ_OQ>(l1, l2);
        _mm256_and_ps(mask, _mm256_set1_ps(1.0))
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm256_cmp_ps::<_CMP_NEQ_UQ>(l1, l2);
        _mm256_and_ps(mask, _mm256_set1_ps(1.0))
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm256_cmp_ps::<_CMP_LT_OQ>(l1, l2);
        _mm256_and_ps(mask, _mm256_set1_ps(1.0))
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm256_cmp_ps::<_CMP_LE_OQ>(l1, l2);
        _mm256_and_ps(mask, _mm256_set1_ps(1.0))
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm256_cmp_ps::<_CMP_GT_OQ>(l1, l2);
        _mm256_and_ps(mask, _mm256_set1_ps(1.0))
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm256_cmp_ps::<_CMP_GE_OQ>(l1, l2);
        _mm256_and_ps(mask, _mm256_set1_ps(1.0))
    }

    #[inline(always)]
    unsafe fn fmadd_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
        acc: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        // A non-fused variant for AVX2 without the FMA cpu feature requirement.
        let res = <Self as SimdRegister<f32>>::mul_dense(l1, l2);
        <Self as SimdRegister<f32>>::add_dense(res, acc)
    }

    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> f32 {
        let left_half = _mm256_extractf128_ps::<1>(reg);
        let right_half = _mm256_castps256_ps128(reg);
        let sum_quad = _mm_add_ps(left_half, right_half);

        let left_half = sum_quad;
        let right_half = _mm_movehl_ps(sum_quad, sum_quad);
        let sum_dual = _mm_add_ps(left_half, right_half);

        let left_half = sum_dual;
        let right_half = _mm_shuffle_ps::<0x1>(sum_dual, sum_dual);
        let sum = _mm_add_ss(left_half, right_half);

        _mm_cvtss_f32(sum)
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> f32 {
        let hi = _mm256_extractf128_ps::<1>(reg);
        let lo = _mm256_castps256_ps128(reg);

        let maxed = _mm_max_ps(lo, hi);
        let [a, b, c, d] = mem::transmute::<__m128, [f32; 4]>(maxed);

        let m1 = a.max(b);
        let m2 = c.max(d);

        m1.max(m2)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> f32 {
        let hi = _mm256_extractf128_ps::<1>(reg);
        let lo = _mm256_castps256_ps128(reg);

        let maxed = _mm_min_ps(lo, hi);
        let [a, b, c, d] = mem::transmute::<_, [f32; 4]>(maxed);

        let m1 = a.min(b);
        let m2 = c.min(d);

        m1.min(m2)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut f32, reg: Self::Register) {
        _mm256_storeu_ps(mem, reg)
    }
}

impl Hypot<f32> for Avx2 {
    //b' * sqrt((a*b_hat)^2 + (b*b_hat)^2), where |b| > |a|, b'=(2^n<b), b_hat=1/b'
    #[inline(always)]
    unsafe fn hypot(x: Self::Register, y: Self::Register) -> Self::Register {
        // convert the inputs to absolute values
        let x = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);

        let y = _mm256_andnot_ps(_mm256_set1_ps(-0.0), y);

        // find the max and min of the two inputs
        let hi = <Avx2 as SimdRegister<f32>>::max(x, y);
        let lo = <Avx2 as SimdRegister<f32>>::min(x, y);

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
        let hi2p = _mm256_blendv_ps(_mm256_set1_ps(f32::MIN_POSITIVE), hi2p, mask);
        // replace the subnormal values of scale with the reciprocal of the minimum positive normal value
        let scale =
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
        let lo_scaled = _mm256_mul_ps(lo, scale);
        let lo_scaled = _mm256_mul_ps(lo_scaled, lo_scaled);
        _mm256_mul_ps(hi2p, _mm256_sqrt_ps(_mm256_add_ps(lo_scaled, hi_scaled)))
    }
}

impl SimdRegister<f64> for Avx2 {
    type Register = __m256d;

    #[inline(always)]
    unsafe fn load(mem: *const f64) -> Self::Register {
        _mm256_loadu_pd(mem)
    }

    #[inline(always)]
    unsafe fn filled(value: f64) -> Self::Register {
        _mm256_set1_pd(value)
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
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_mul_pd(l1, l2)
    }

    #[inline(always)]
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_div_pd(l1, l2)
    }

    #[inline(always)]
    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        // A non-fused variant for AVX2 without the FMA cpu feature requirement.
        let res = <Self as SimdRegister<f64>>::mul(l1, l2);
        <Self as SimdRegister<f64>>::add(res, acc)
    }

    #[inline(always)]
    unsafe fn max(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_max_pd(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_min_pd(l1, l2)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm256_cmp_pd::<_CMP_EQ_OQ>(l1, l2);
        _mm256_and_pd(mask, _mm256_set1_pd(1.0))
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm256_cmp_pd::<_CMP_NEQ_UQ>(l1, l2);
        _mm256_and_pd(mask, _mm256_set1_pd(1.0))
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm256_cmp_pd::<_CMP_LT_OQ>(l1, l2);
        _mm256_and_pd(mask, _mm256_set1_pd(1.0))
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm256_cmp_pd::<_CMP_LE_OQ>(l1, l2);
        _mm256_and_pd(mask, _mm256_set1_pd(1.0))
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm256_cmp_pd::<_CMP_GT_OQ>(l1, l2);
        _mm256_and_pd(mask, _mm256_set1_pd(1.0))
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm256_cmp_pd::<_CMP_GE_OQ>(l1, l2);
        _mm256_and_pd(mask, _mm256_set1_pd(1.0))
    }

    #[inline(always)]
    unsafe fn fmadd_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
        acc: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        // A non-fused variant for AVX2 without the FMA cpu feature requirement.
        let res = <Self as SimdRegister<f64>>::mul_dense(l1, l2);
        <Self as SimdRegister<f64>>::add_dense(res, acc)
    }

    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> f64 {
        let left_half = _mm256_extractf128_pd::<1>(reg);
        let right_half = _mm256_castpd256_pd128(reg);
        let sum_duo = _mm_add_pd(left_half, right_half);

        let undef = _mm_undefined_ps();
        let shuffle_tmp = _mm_movehl_ps(undef, _mm_castpd_ps(sum_duo));
        let shuffle = _mm_castps_pd(shuffle_tmp);
        _mm_cvtsd_f64(_mm_add_sd(sum_duo, shuffle))
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> f64 {
        let hi = _mm256_extractf128_pd::<1>(reg);
        let lo = _mm256_castpd256_pd128(reg);

        let maxed = _mm_max_pd(lo, hi);
        let [a, b] = mem::transmute::<_, [f64; 2]>(maxed);

        a.max(b)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> f64 {
        let hi = _mm256_extractf128_pd::<1>(reg);
        let lo = _mm256_castpd256_pd128(reg);

        let maxed = _mm_min_pd(lo, hi);
        let [a, b] = mem::transmute::<_, [f64; 2]>(maxed);

        a.min(b)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut f64, reg: Self::Register) {
        _mm256_storeu_pd(mem, reg)
    }
}

impl SimdRegister<i8> for Avx2 {
    type Register = __m256i;

    #[inline(always)]
    unsafe fn load(mem: *const i8) -> Self::Register {
        _mm256_loadu_si256(mem.cast())
    }

    #[inline(always)]
    unsafe fn filled(value: i8) -> Self::Register {
        _mm256_set1_epi8(value)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        _mm256_setzero_si256()
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_add_epi8(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_sub_epi8(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm256_set1_epi32(0xFF00FF00u32 as i32);

        let shift_l1 = _mm256_srai_epi16::<8>(l1);
        let shift_l2 = _mm256_srai_epi16::<8>(l2);

        let even = _mm256_mullo_epi16(l1, l2);
        let odd = _mm256_mullo_epi16(shift_l1, shift_l2);
        let odd = _mm256_slli_epi16::<8>(odd);
        _mm256_blendv_epi8(even, odd, mask)
    }

    #[inline(always)]
    /// Scalar `i8` integer division.
    ///
    /// In reality this operation is not SIMD, in theory we could later support
    /// it however it will always be an incredibly expensive operation with quite
    /// a lot of cognitive load on the maintenance side, so for the foreseeable future
    /// non-floating point division operations will be non-simd.
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let l1_unpacked = mem::transmute::<_, [i8; 32]>(l1);
        let l2_unpacked = mem::transmute::<_, [i8; 32]>(l2);

        let mut result = [0i8; 32];
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
        let res = <Self as SimdRegister<i8>>::mul(l1, l2);
        <Self as SimdRegister<i8>>::add(res, acc)
    }

    #[inline(always)]
    unsafe fn max(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_max_epi8(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_min_epi8(l1, l2)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm256_cmpeq_epi8(l1, l2);
        _mm256_and_si256(mask, _mm256_set1_epi8(1))
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let eq_mask = _mm256_cmpeq_epi8(l1, l2);
        _mm256_andnot_si256(eq_mask, _mm256_set1_epi8(1))
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Self as SimdRegister<i8>>::gt(l2, l1)
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Self as SimdRegister<i8>>::gte(l2, l1)
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm256_cmpgt_epi8(l1, l2);
        _mm256_and_si256(mask, _mm256_set1_epi8(1))
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let swapped_cmp = _mm256_cmpgt_epi8(l2, l1);
        _mm256_andnot_si256(swapped_cmp, _mm256_set1_epi8(1))
    }

    #[inline(always)]
    unsafe fn mul_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let mask = DenseLane::copy(_mm256_set1_epi32(0xFF00FF00u32 as i32));

        let even = apply_dense!(_mm256_mullo_epi16, l1, l2);

        let shift_l1 = apply_dense!(_mm256_srai_epi16::<8>, l1);
        let shift_l2 = apply_dense!(_mm256_srai_epi16::<8>, l2);

        let odd = apply_dense!(_mm256_mullo_epi16, shift_l1, shift_l2);
        let odd = apply_dense!(_mm256_slli_epi16::<8>, odd);

        apply_dense!(_mm256_blendv_epi8, even, odd, mask)
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
    unsafe fn eq_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let mask = apply_dense!(_mm256_cmpeq_epi8, l1, l2);
        apply_dense!(_mm256_and_si256, mask, value = _mm256_set1_epi8(1))
    }

    #[inline(always)]
    unsafe fn neq_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let mask = apply_dense!(_mm256_cmpeq_epi8, l1, l2);
        apply_dense!(_mm256_andnot_si256, mask, value = _mm256_set1_epi8(1))
    }

    #[inline(always)]
    unsafe fn lt_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        <Self as SimdRegister<i8>>::gt_dense(l2, l1)
    }

    #[inline(always)]
    unsafe fn lte_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        <Self as SimdRegister<i8>>::gte_dense(l2, l1)
    }

    #[inline(always)]
    unsafe fn gt_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let mask = apply_dense!(_mm256_cmpgt_epi8, l1, l2);
        apply_dense!(_mm256_and_si256, mask, value = _mm256_set1_epi8(1))
    }

    #[inline(always)]
    unsafe fn gte_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let swapped_cmp = apply_dense!(_mm256_cmpgt_epi8, l2, l1);
        apply_dense!(
            _mm256_andnot_si256,
            swapped_cmp,
            value = _mm256_set1_epi8(1)
        )
    }

    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> i8 {
        // There is a bit of an assumption the compile will optimize these scalar impls
        // out, but the SIMD version is a bit complicated and is difficult to get to
        // mirror the scalar behaviour.
        // TODO: We can probably do this with the pure SIMD way, but we should try match
        //       the scalar behaviour in terms of wrapping.

        let hi = _mm256_extracti128_si256::<1>(reg);
        let lo = _mm256_castsi256_si128(reg);

        let sum = _mm_add_epi8(hi, lo);
        let unpacked = mem::transmute::<_, [i8; 16]>(sum);

        let mut s1: i8 = 0;
        let mut s2: i8 = 0;
        let mut s3: i8 = 0;
        let mut s4: i8 = 0;

        let mut i = 0;
        while i < 16 {
            s1 = s1.wrapping_add(unpacked[i]);
            s2 = s2.wrapping_add(unpacked[i + 1]);
            s3 = s3.wrapping_add(unpacked[i + 2]);
            s4 = s4.wrapping_add(unpacked[i + 3]);

            i += 4;
        }

        s1 = s1.wrapping_add(s2);
        s3 = s3.wrapping_add(s4);

        s1.wrapping_add(s3)
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> i8 {
        let hi = _mm256_extracti128_si256::<1>(reg);
        let lo = _mm256_castsi256_si128(reg);

        let maxed = _mm_max_epi8(hi, lo);
        let unpacked = mem::transmute::<_, [i8; 16]>(maxed);

        let mut m1 = i8::MIN;
        let mut m2 = i8::MIN;
        let mut m3 = i8::MIN;
        let mut m4 = i8::MIN;

        let mut i = 0;
        while i < 16 {
            m1 = m1.max(unpacked[i]);
            m2 = m2.max(unpacked[i + 1]);
            m3 = m3.max(unpacked[i + 2]);
            m4 = m4.max(unpacked[i + 3]);

            i += 4;
        }

        m1 = m1.max(m2);
        m3 = m3.max(m4);

        m1.max(m3)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> i8 {
        let hi = _mm256_extracti128_si256::<1>(reg);
        let lo = _mm256_castsi256_si128(reg);

        let minimal = _mm_min_epi8(hi, lo);
        let unpacked = mem::transmute::<_, [i8; 16]>(minimal);

        let mut m1 = i8::MAX;
        let mut m2 = i8::MAX;
        let mut m3 = i8::MAX;
        let mut m4 = i8::MAX;

        let mut i = 0;
        while i < 16 {
            m1 = m1.min(unpacked[i]);
            m2 = m2.min(unpacked[i + 1]);
            m3 = m3.min(unpacked[i + 2]);
            m4 = m4.min(unpacked[i + 3]);

            i += 4;
        }

        m1 = m1.min(m2);
        m3 = m3.min(m4);

        m1.min(m3)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut i8, reg: Self::Register) {
        _mm256_storeu_si256(mem.cast(), reg)
    }
}

impl SimdRegister<i16> for Avx2 {
    type Register = __m256i;

    #[inline(always)]
    unsafe fn load(mem: *const i16) -> Self::Register {
        _mm256_loadu_si256(mem.cast())
    }

    #[inline(always)]
    unsafe fn filled(value: i16) -> Self::Register {
        _mm256_set1_epi16(value)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        _mm256_setzero_si256()
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_add_epi16(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_sub_epi16(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_mullo_epi16(l1, l2)
    }

    #[inline(always)]
    /// Scalar `i16` integer division.
    ///
    /// In reality this operation is not SIMD, in theory we could later support
    /// it however it will always be an incredibly expensive operation with quite
    /// a lot of cognitive load on the maintenance side, so for the foreseeable future
    /// non-floating point division operations will be non-simd.
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let l1_unpacked = mem::transmute::<_, [i16; 16]>(l1);
        let l2_unpacked = mem::transmute::<_, [i16; 16]>(l2);

        let mut result = [0i16; 16];
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
        _mm256_max_epi16(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_min_epi16(l1, l2)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm256_cmpeq_epi16(l1, l2);
        _mm256_and_si256(mask, _mm256_set1_epi16(1))
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let eq_mask = _mm256_cmpeq_epi16(l1, l2);
        _mm256_andnot_si256(eq_mask, _mm256_set1_epi16(1))
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Self as SimdRegister<i16>>::gt(l2, l1)
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Self as SimdRegister<i16>>::gte(l2, l1)
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm256_cmpgt_epi16(l1, l2);
        // Small optimization for 16, 32 and 64bit values which
        // can shift instead of doing a bitwise `and` on a mask
        _mm256_srli_epi16::<15>(mask)
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let swapped_cmp = _mm256_cmpgt_epi16(l2, l1);
        let mask = _mm256_xor_si256(swapped_cmp, _mm256_set1_epi16(-1));
        // Small optimization for 16, 32 and 64bit values which
        // can shift instead of doing a bitwise `and` on a mask
        _mm256_srli_epi16::<15>(mask)
    }

    #[inline(always)]
    unsafe fn mul_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        apply_dense!(_mm256_mullo_epi16, l1, l2)
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
    unsafe fn eq_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let mask = apply_dense!(_mm256_cmpeq_epi16, l1, l2);
        apply_dense!(_mm256_srli_epi16::<15>, mask)
    }

    #[inline(always)]
    unsafe fn neq_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let mask = apply_dense!(_mm256_cmpeq_epi16, l1, l2);
        apply_dense!(_mm256_andnot_si256, mask, value = _mm256_set1_epi16(1))
    }

    #[inline(always)]
    unsafe fn lt_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        <Self as SimdRegister<i16>>::gt_dense(l2, l1)
    }

    #[inline(always)]
    unsafe fn lte_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        <Self as SimdRegister<i16>>::gte_dense(l2, l1)
    }

    #[inline(always)]
    unsafe fn gt_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let mask = apply_dense!(_mm256_cmpgt_epi16, l1, l2);
        apply_dense!(_mm256_srli_epi16::<15>, mask)
    }

    #[inline(always)]
    unsafe fn gte_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let swapped_cmp = apply_dense!(_mm256_cmpgt_epi16, l2, l1);
        apply_dense!(
            _mm256_andnot_si256,
            swapped_cmp,
            value = _mm256_set1_epi16(1)
        )
    }

    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> i16 {
        // There is a bit of an assumption the compile will optimize these scalar impls
        // out, but the SIMD version is a bit complicated and is difficult to get to
        // mirror the scalar behaviour.
        // TODO: We can probably do this with the pure SIMD way, but we should try match
        //       the scalar behaviour in terms of wrapping.

        let hi = _mm256_extracti128_si256::<1>(reg);
        let lo = _mm256_castsi256_si128(reg);

        let sum = _mm_add_epi16(hi, lo);
        let unpacked = mem::transmute::<_, [i16; 8]>(sum);

        let mut s1: i16 = 0;
        let mut s2: i16 = 0;
        let mut s3: i16 = 0;
        let mut s4: i16 = 0;

        let mut i = 0;
        while i < 8 {
            s1 = s1.wrapping_add(unpacked[i]);
            s2 = s2.wrapping_add(unpacked[i + 1]);
            s3 = s3.wrapping_add(unpacked[i + 2]);
            s4 = s4.wrapping_add(unpacked[i + 3]);

            i += 4;
        }

        s1 = s1.wrapping_add(s2);
        s3 = s3.wrapping_add(s4);

        s1.wrapping_add(s3)
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> i16 {
        let hi = _mm256_extracti128_si256::<1>(reg);
        let lo = _mm256_castsi256_si128(reg);

        let maxed = _mm_max_epi16(hi, lo);
        let unpacked = mem::transmute::<_, [i16; 8]>(maxed);

        let mut m1 = i16::MIN;
        let mut m2 = i16::MIN;
        let mut m3 = i16::MIN;
        let mut m4 = i16::MIN;

        let mut i = 0;
        while i < 8 {
            m1 = m1.max(unpacked[i]);
            m2 = m2.max(unpacked[i + 1]);
            m3 = m3.max(unpacked[i + 2]);
            m4 = m4.max(unpacked[i + 3]);

            i += 4;
        }

        m1 = m1.max(m2);
        m3 = m3.max(m4);

        m1.max(m3)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> i16 {
        let hi = _mm256_extracti128_si256::<1>(reg);
        let lo = _mm256_castsi256_si128(reg);

        let minimal = _mm_min_epi16(hi, lo);
        let unpacked = mem::transmute::<_, [i16; 8]>(minimal);

        let mut m1 = i16::MAX;
        let mut m2 = i16::MAX;
        let mut m3 = i16::MAX;
        let mut m4 = i16::MAX;

        let mut i = 0;
        while i < 8 {
            m1 = m1.min(unpacked[i]);
            m2 = m2.min(unpacked[i + 1]);
            m3 = m3.min(unpacked[i + 2]);
            m4 = m4.min(unpacked[i + 3]);

            i += 4;
        }

        m1 = m1.min(m2);
        m3 = m3.min(m4);

        m1.min(m3)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut i16, reg: Self::Register) {
        _mm256_storeu_si256(mem.cast(), reg)
    }
}

impl SimdRegister<i32> for Avx2 {
    type Register = __m256i;

    #[inline(always)]
    unsafe fn load(mem: *const i32) -> Self::Register {
        _mm256_loadu_si256(mem.cast())
    }

    #[inline(always)]
    unsafe fn filled(value: i32) -> Self::Register {
        _mm256_set1_epi32(value)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        _mm256_setzero_si256()
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_add_epi32(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_sub_epi32(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_mullo_epi32(l1, l2)
    }

    #[inline(always)]
    /// Scalar `i32` integer division.
    ///
    /// In reality this operation is not SIMD, in theory we could later support
    /// it however it will always be an incredibly expensive operation with quite
    /// a lot of cognitive load on the maintenance side, so for the foreseeable future
    /// non-floating point division operations will be non-simd.
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let l1_unpacked = mem::transmute::<_, [i32; 8]>(l1);
        let l2_unpacked = mem::transmute::<_, [i32; 8]>(l2);

        let mut result = [0i32; 8];
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
        _mm256_max_epi32(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_min_epi32(l1, l2)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm256_cmpeq_epi32(l1, l2);
        _mm256_and_si256(mask, _mm256_set1_epi32(1))
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let eq_mask = _mm256_cmpeq_epi32(l1, l2);
        _mm256_andnot_si256(eq_mask, _mm256_set1_epi32(1))
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Self as SimdRegister<i32>>::gt(l2, l1)
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Self as SimdRegister<i32>>::gte(l2, l1)
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm256_cmpgt_epi32(l1, l2);
        // Small optimization for 16, 32 and 64bit values which
        // can shift instead of doing a bitwise `and` on a mask
        _mm256_srli_epi32::<31>(mask)
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let swapped_cmp = _mm256_cmpgt_epi32(l2, l1);
        let mask = _mm256_xor_si256(swapped_cmp, _mm256_set1_epi32(-1));
        // Small optimization for 16, 32 and 64bit values which
        // can shift instead of doing a bitwise `and` on a mask
        _mm256_srli_epi32::<31>(mask)
    }

    #[inline(always)]
    unsafe fn mul_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        apply_dense!(_mm256_mullo_epi32, l1, l2)
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
    unsafe fn eq_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let mask = apply_dense!(_mm256_cmpeq_epi32, l1, l2);
        apply_dense!(_mm256_srli_epi32::<31>, mask)
    }

    #[inline(always)]
    unsafe fn neq_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let mask = apply_dense!(_mm256_cmpeq_epi32, l1, l2);
        apply_dense!(_mm256_andnot_si256, mask, value = _mm256_set1_epi32(1))
    }

    #[inline(always)]
    unsafe fn lt_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        <Self as SimdRegister<i32>>::gt_dense(l2, l1)
    }

    #[inline(always)]
    unsafe fn lte_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        <Self as SimdRegister<i32>>::gte_dense(l2, l1)
    }

    #[inline(always)]
    unsafe fn gt_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let mask = apply_dense!(_mm256_cmpgt_epi32, l1, l2);
        apply_dense!(_mm256_srli_epi32::<31>, mask)
    }

    #[inline(always)]
    unsafe fn gte_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let swapped_cmp = apply_dense!(_mm256_cmpgt_epi32, l2, l1);
        apply_dense!(
            _mm256_andnot_si256,
            swapped_cmp,
            value = _mm256_set1_epi32(1)
        )
    }

    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> i32 {
        // There is a bit of an assumption the compile will optimize these scalar impls
        // out, but the SIMD version is a bit complicated and is difficult to get to
        // mirror the scalar behaviour.
        // TODO: We can probably do this with the pure SIMD way, but we should try match
        //       the scalar behaviour in terms of wrapping.

        let hi = _mm256_extracti128_si256::<1>(reg);
        let lo = _mm256_castsi256_si128(reg);

        let sum = _mm_add_epi32(hi, lo);
        let unpacked = mem::transmute::<_, [i32; 4]>(sum);

        let mut s1 = unpacked[0];
        let s2 = unpacked[1];
        let mut s3 = unpacked[2];
        let s4 = unpacked[3];

        s1 = s1.wrapping_add(s2);
        s3 = s3.wrapping_add(s4);

        s1.wrapping_add(s3)
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> i32 {
        let hi = _mm256_extracti128_si256::<1>(reg);
        let lo = _mm256_castsi256_si128(reg);

        let maxed = _mm_max_epi32(hi, lo);
        let unpacked = mem::transmute::<_, [i32; 4]>(maxed);

        let mut m1 = unpacked[0];
        let m2 = unpacked[1];
        let mut m3 = unpacked[2];
        let m4 = unpacked[3];

        m1 = m1.max(m2);
        m3 = m3.max(m4);

        m1.max(m3)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> i32 {
        let hi = _mm256_extracti128_si256::<1>(reg);
        let lo = _mm256_castsi256_si128(reg);

        let minimal = _mm_min_epi32(hi, lo);
        let unpacked = mem::transmute::<_, [i32; 4]>(minimal);

        let mut m1 = unpacked[0];
        let m2 = unpacked[1];
        let mut m3 = unpacked[2];
        let m4 = unpacked[3];

        m1 = m1.min(m2);
        m3 = m3.min(m4);

        m1.min(m3)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut i32, reg: Self::Register) {
        _mm256_storeu_si256(mem.cast(), reg)
    }
}

impl SimdRegister<i64> for Avx2 {
    type Register = __m256i;

    #[inline(always)]
    unsafe fn load(mem: *const i64) -> Self::Register {
        _mm256_loadu_si256(mem.cast())
    }

    #[inline(always)]
    unsafe fn filled(value: i64) -> Self::Register {
        _mm256_set1_epi64x(value)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        _mm256_setzero_si256()
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_add_epi64(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_sub_epi64(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm256_set1_epi64x(0xFFFFFFFF00000000u64 as i64);
        let digit_1 = _mm256_mul_epu32(l1, l2);

        let l2_swap = _mm256_shuffle_epi32::<{ super::_MM_SHUFFLE(2, 3, 0, 1) }>(l2);
        let cross_prod = _mm256_mullo_epi32(l1, l2_swap);

        let prod_lo = _mm256_slli_epi64::<32>(cross_prod);
        let sum_cross = _mm256_add_epi32(prod_lo, cross_prod);
        let digit_2 = _mm256_and_si256(sum_cross, mask);

        _mm256_add_epi64(digit_1, digit_2)
    }

    #[inline(always)]
    /// Scalar `i64` integer division.
    ///
    /// In reality this operation is not SIMD, in theory we could later support
    /// it however it will always be an incredibly expensive operation with quite
    /// a lot of cognitive load on the maintenance side, so for the foreseeable future
    /// non-floating point division operations will be non-simd.
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let l1_unpacked = mem::transmute::<_, [i64; 4]>(l1);
        let l2_unpacked = mem::transmute::<_, [i64; 4]>(l2);

        let mut result = [0i64; 4];
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
        let mask = _mm256_cmpgt_epi64(l1, l2);
        _mm256_blendv_epi8(l2, l1, mask)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm256_cmpgt_epi64(l1, l2);
        _mm256_blendv_epi8(l1, l2, mask)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm256_cmpeq_epi64(l1, l2);
        _mm256_srli_epi64::<63>(mask)
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let eq_mask = _mm256_cmpeq_epi64(l1, l2);
        _mm256_andnot_si256(eq_mask, _mm256_set1_epi64x(1))
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Self as SimdRegister<i64>>::gt(l2, l1)
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Self as SimdRegister<i64>>::gte(l2, l1)
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm256_cmpgt_epi64(l1, l2);
        // Small optimization for 16, 32 and 64bit values which
        // can shift instead of doing a bitwise `and` on a mask
        _mm256_srli_epi64::<63>(mask)
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let swapped_cmp = _mm256_cmpgt_epi64(l2, l1);
        // Because we have to do a bitwise not using a broadcast value, we can
        // cheat and just use andnot as a fused operation for also converting our mask
        _mm256_andnot_si256(swapped_cmp, _mm256_set1_epi64x(1))
    }

    #[inline(always)]
    unsafe fn mul_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let mask = DenseLane::copy(_mm256_set1_epi64x(0xFFFFFFFF00000000u64 as i64));

        let digit_1 = apply_dense!(_mm256_mul_epu32, l1, l2);

        let l2_swap = apply_dense!(
            _mm256_shuffle_epi32::<{ super::_MM_SHUFFLE(2, 3, 0, 1) }>,
            l2
        );
        let cross_prod = apply_dense!(_mm256_mullo_epi32, l1, l2_swap);

        let prod_lo = apply_dense!(_mm256_slli_epi64::<32>, cross_prod);
        let sum_cross = apply_dense!(_mm256_add_epi32, prod_lo, cross_prod);
        let digit_2 = apply_dense!(_mm256_and_si256, sum_cross, mask);

        apply_dense!(_mm256_add_epi64, digit_1, digit_2)
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
    unsafe fn eq_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let mask = apply_dense!(_mm256_cmpeq_epi64, l1, l2);
        apply_dense!(_mm256_srli_epi64::<63>, mask)
    }

    #[inline(always)]
    unsafe fn neq_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let mask = apply_dense!(_mm256_cmpeq_epi64, l1, l2);
        apply_dense!(_mm256_andnot_si256, mask, value = _mm256_set1_epi64x(1))
    }

    #[inline(always)]
    unsafe fn lt_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        <Self as SimdRegister<i64>>::gt_dense(l2, l1)
    }

    #[inline(always)]
    unsafe fn lte_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        <Self as SimdRegister<i64>>::gte_dense(l2, l1)
    }

    #[inline(always)]
    unsafe fn gt_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let mask = apply_dense!(_mm256_cmpgt_epi64, l1, l2);
        apply_dense!(_mm256_srli_epi64::<63>, mask)
    }

    #[inline(always)]
    unsafe fn gte_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let swapped_cmp = apply_dense!(_mm256_cmpgt_epi64, l2, l1);
        apply_dense!(
            _mm256_andnot_si256,
            swapped_cmp,
            value = _mm256_set1_epi64x(1)
        )
    }

    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> i64 {
        // There is a bit of an assumption the compile will optimize these scalar impls
        // out, but the SIMD version is a bit complicated and is difficult to get to
        // mirror the scalar behaviour.
        // TODO: We can probably do this with the pure SIMD way, but we should try match
        //       the scalar behaviour in terms of wrapping.

        let hi = _mm256_extracti128_si256::<1>(reg);
        let lo = _mm256_castsi256_si128(reg);

        let sum = _mm_add_epi64(hi, lo);
        let unpacked = mem::transmute::<_, [i64; 2]>(sum);

        let s1 = unpacked[0];
        let s2 = unpacked[1];

        s1.wrapping_add(s2)
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> i64 {
        let hi = _mm256_extracti128_si256::<1>(reg);
        let lo = _mm256_castsi256_si128(reg);

        let mask = _mm_cmpgt_epi64(hi, lo);
        let max = _mm_blendv_epi8(lo, hi, mask);

        let [m1, m2] = mem::transmute::<_, [i64; 2]>(max);

        m1.max(m2)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> i64 {
        let hi = _mm256_extracti128_si256::<1>(reg);
        let lo = _mm256_castsi256_si128(reg);

        let mask = _mm_cmpgt_epi64(hi, lo);
        let min = _mm_blendv_epi8(hi, lo, mask);

        let [m1, m2] = mem::transmute::<_, [i64; 2]>(min);

        m1.min(m2)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut i64, reg: Self::Register) {
        _mm256_storeu_si256(mem.cast(), reg)
    }
}

impl SimdRegister<u8> for Avx2 {
    type Register = __m256i;

    #[inline(always)]
    unsafe fn load(mem: *const u8) -> Self::Register {
        _mm256_loadu_si256(mem.cast())
    }

    #[inline(always)]
    unsafe fn filled(value: u8) -> Self::Register {
        _mm256_set1_epi8(value as i8)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        _mm256_setzero_si256()
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_add_epi8(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_sub_epi8(l1, l2)
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
        let l1_unpacked = mem::transmute::<_, [u8; 32]>(l1);
        let l2_unpacked = mem::transmute::<_, [u8; 32]>(l2);

        let mut result = [0u8; 32];
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
        _mm256_max_epu8(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_min_epu8(l1, l2)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Self as SimdRegister<i8>>::eq(l1, l2) // Operation is identical
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Self as SimdRegister<i8>>::neq(l1, l2) // Operation is identical
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Self as SimdRegister<u8>>::gt(l2, l1)
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Self as SimdRegister<u8>>::gte(l2, l1)
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let sign = _mm256_set1_epi32(0x80808080u32 as i32);
        let l1_xor = _mm256_xor_si256(l1, sign);
        let l2_xor = _mm256_xor_si256(l2, sign);
        let mask = _mm256_cmpgt_epi8(l1_xor, l2_xor);
        _mm256_and_si256(mask, _mm256_set1_epi8(1))
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm256_cmpeq_epi8(l1, _mm256_max_epu8(l1, l2));
        _mm256_and_si256(mask, _mm256_set1_epi8(1))
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
    unsafe fn eq_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        <Self as SimdRegister<i8>>::eq_dense(l1, l2)
    }

    #[inline(always)]
    unsafe fn neq_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        <Self as SimdRegister<i8>>::neq_dense(l1, l2)
    }

    #[inline(always)]
    unsafe fn lt_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        <Self as SimdRegister<u8>>::gt_dense(l2, l1)
    }

    #[inline(always)]
    unsafe fn lte_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        <Self as SimdRegister<u8>>::gte_dense(l2, l1)
    }

    #[inline(always)]
    unsafe fn gt_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let sign = _mm256_set1_epi32(0x80_80_80_80u32 as i32);

        // We have to split the dense lane operations up into specific parts
        // because otherwise we run out of registers. Here we target no more than 16
        // registers in use.
        //
        // Do the 1st quarter of the dense lane
        let l1_a_xor = _mm256_xor_si256(l1.a, sign);
        let l1_b_xor = _mm256_xor_si256(l1.b, sign);
        let l2_a_xor = _mm256_xor_si256(l2.a, sign);
        let l2_b_xor = _mm256_xor_si256(l2.b, sign);
        let mask_a = _mm256_cmpgt_epi8(l1_a_xor, l2_a_xor);
        let mask_b = _mm256_cmpgt_epi8(l1_b_xor, l2_b_xor);

        // Do the 2nd quarter of the dense lane
        let l2_c_xor = _mm256_xor_si256(l2.c, sign);
        let l2_d_xor = _mm256_xor_si256(l2.d, sign);
        let l1_c_xor = _mm256_xor_si256(l1.c, sign);
        let l1_d_xor = _mm256_xor_si256(l1.d, sign);
        let mask_c = _mm256_cmpgt_epi8(l1_c_xor, l2_c_xor);
        let mask_d = _mm256_cmpgt_epi8(l1_d_xor, l2_d_xor);

        // Do the 3rd quarter
        let l1_e_xor = _mm256_xor_si256(l1.e, sign);
        let l1_f_xor = _mm256_xor_si256(l1.f, sign);
        let l2_e_xor = _mm256_xor_si256(l2.e, sign);
        let l2_f_xor = _mm256_xor_si256(l2.f, sign);
        let mask_e = _mm256_cmpgt_epi8(l1_e_xor, l2_e_xor);
        let mask_f = _mm256_cmpgt_epi8(l1_f_xor, l2_f_xor);

        // Do the 4th quarter
        let l1_g_xor = _mm256_xor_si256(l1.g, sign);
        let l1_h_xor = _mm256_xor_si256(l1.h, sign);
        let l2_g_xor = _mm256_xor_si256(l2.g, sign);
        let l2_h_xor = _mm256_xor_si256(l2.h, sign);
        let mask_g = _mm256_cmpgt_epi8(l1_g_xor, l2_g_xor);
        let mask_h = _mm256_cmpgt_epi8(l1_h_xor, l2_h_xor);

        let mask = DenseLane {
            a: mask_a,
            b: mask_b,
            c: mask_c,
            d: mask_d,
            e: mask_e,
            f: mask_f,
            g: mask_g,
            h: mask_h,
        };

        apply_dense!(_mm256_and_si256, mask, value = _mm256_set1_epi8(1))
    }

    #[inline(always)]
    unsafe fn gte_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let maxed = apply_dense!(_mm256_max_epu8, l1, l2);
        let mask = apply_dense!(_mm256_cmpeq_epi8, l1, maxed);
        apply_dense!(_mm256_and_si256, mask, value = _mm256_set1_epi8(1))
    }

    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> u8 {
        // There is a bit of an assumption the compile will optimize these scalar impls
        // out, but the SIMD version is a bit complicated and is difficult to get to
        // mirror the scalar behaviour.
        // TODO: We can probably do this with the pure SIMD way, but we should try match
        //       the scalar behaviour in terms of wrapping.

        let hi = _mm256_extracti128_si256::<1>(reg);
        let lo = _mm256_castsi256_si128(reg);

        let sum = _mm_add_epi8(hi, lo);
        let unpacked = mem::transmute::<_, [u8; 16]>(sum);

        let mut s1: u8 = 0;
        let mut s2: u8 = 0;
        let mut s3: u8 = 0;
        let mut s4: u8 = 0;

        let mut i = 0;
        while i < 16 {
            s1 = s1.wrapping_add(unpacked[i]);
            s2 = s2.wrapping_add(unpacked[i + 1]);
            s3 = s3.wrapping_add(unpacked[i + 2]);
            s4 = s4.wrapping_add(unpacked[i + 3]);

            i += 4;
        }

        s1 = s1.wrapping_add(s2);
        s3 = s3.wrapping_add(s4);

        s1.wrapping_add(s3)
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> u8 {
        let hi = _mm256_extracti128_si256::<1>(reg);
        let lo = _mm256_castsi256_si128(reg);

        let maxed = _mm_max_epu8(hi, lo);
        let unpacked = mem::transmute::<_, [u8; 16]>(maxed);

        let mut m1 = u8::MIN;
        let mut m2 = u8::MIN;
        let mut m3 = u8::MIN;
        let mut m4 = u8::MIN;

        let mut i = 0;
        while i < 16 {
            m1 = m1.max(unpacked[i]);
            m2 = m2.max(unpacked[i + 1]);
            m3 = m3.max(unpacked[i + 2]);
            m4 = m4.max(unpacked[i + 3]);

            i += 4;
        }

        m1 = m1.max(m2);
        m3 = m3.max(m4);

        m1.max(m3)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> u8 {
        let hi = _mm256_extracti128_si256::<1>(reg);
        let lo = _mm256_castsi256_si128(reg);

        let minimal = _mm_min_epu8(hi, lo);
        let unpacked = mem::transmute::<_, [u8; 16]>(minimal);

        let mut m1 = u8::MAX;
        let mut m2 = u8::MAX;
        let mut m3 = u8::MAX;
        let mut m4 = u8::MAX;

        let mut i = 0;
        while i < 16 {
            m1 = m1.min(unpacked[i]);
            m2 = m2.min(unpacked[i + 1]);
            m3 = m3.min(unpacked[i + 2]);
            m4 = m4.min(unpacked[i + 3]);

            i += 4;
        }

        m1 = m1.min(m2);
        m3 = m3.min(m4);

        m1.min(m3)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut u8, reg: Self::Register) {
        _mm256_storeu_si256(mem.cast(), reg)
    }
}

impl SimdRegister<u16> for Avx2 {
    type Register = __m256i;

    #[inline(always)]
    unsafe fn load(mem: *const u16) -> Self::Register {
        _mm256_loadu_si256(mem.cast())
    }

    #[inline(always)]
    unsafe fn filled(value: u16) -> Self::Register {
        _mm256_set1_epi16(value as i16)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        _mm256_setzero_si256()
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_add_epi16(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_sub_epi16(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_mullo_epi16(l1, l2)
    }

    #[inline(always)]
    /// Scalar `u16` integer division.
    ///
    /// In reality this operation is not SIMD, in theory we could later support
    /// it however it will always be an incredibly expensive operation with quite
    /// a lot of cognitive load on the maintenance side, so for the foreseeable future
    /// non-floating point division operations will be non-simd.
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let l1_unpacked = mem::transmute::<_, [u16; 16]>(l1);
        let l2_unpacked = mem::transmute::<_, [u16; 16]>(l2);

        let mut result = [0u16; 16];
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
        let res = <Self as SimdRegister<u16>>::mul(l1, l2);
        <Self as SimdRegister<u16>>::add(res, acc)
    }

    #[inline(always)]
    unsafe fn max(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_max_epu16(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_min_epu16(l1, l2)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Self as SimdRegister<i16>>::eq(l1, l2) // Operation is identical
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Self as SimdRegister<i16>>::neq(l1, l2) // Operation is identical
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Self as SimdRegister<u16>>::gt(l2, l1)
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Self as SimdRegister<u16>>::gte(l2, l1)
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let sign = _mm256_set1_epi32(0x8000_8000u32 as i32);
        let l1_xor = _mm256_xor_si256(l1, sign);
        let l2_xor = _mm256_xor_si256(l2, sign);
        let mask = _mm256_cmpgt_epi16(l1_xor, l2_xor);
        // Small optimization for 16, 32 and 64bit values which
        // can shift instead of doing a bitwise `and` on a mask
        _mm256_srli_epi16::<15>(mask)
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm256_cmpeq_epi16(l1, _mm256_max_epu16(l1, l2));
        // Small optimization for 16, 32 and 64bit values which
        // can shift instead of doing a bitwise `and` on a mask
        _mm256_srli_epi16::<15>(mask)
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
    unsafe fn eq_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        <Self as SimdRegister<i16>>::eq_dense(l1, l2)
    }

    #[inline(always)]
    unsafe fn neq_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        <Self as SimdRegister<i16>>::neq_dense(l1, l2)
    }

    #[inline(always)]
    unsafe fn lt_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        <Self as SimdRegister<u16>>::gt_dense(l2, l1)
    }

    #[inline(always)]
    unsafe fn lte_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        <Self as SimdRegister<u16>>::gte_dense(l2, l1)
    }

    #[inline(always)]
    unsafe fn gt_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let sign = _mm256_set1_epi32(0x80008000u32 as i32);

        // We have to split the dense lane operations up into specific parts
        // because otherwise we run out of registers. Here we target no more than 16
        // registers in use.
        //
        // Do the 1st quarter of the dense lane
        let l1_a_xor = _mm256_xor_si256(l1.a, sign);
        let l1_b_xor = _mm256_xor_si256(l1.b, sign);
        let l2_a_xor = _mm256_xor_si256(l2.a, sign);
        let l2_b_xor = _mm256_xor_si256(l2.b, sign);
        let mask_a = _mm256_cmpgt_epi16(l1_a_xor, l2_a_xor);
        let mask_b = _mm256_cmpgt_epi16(l1_b_xor, l2_b_xor);

        // Do the 2nd quarter of the dense lane
        let l2_c_xor = _mm256_xor_si256(l2.c, sign);
        let l2_d_xor = _mm256_xor_si256(l2.d, sign);
        let l1_c_xor = _mm256_xor_si256(l1.c, sign);
        let l1_d_xor = _mm256_xor_si256(l1.d, sign);
        let mask_c = _mm256_cmpgt_epi16(l1_c_xor, l2_c_xor);
        let mask_d = _mm256_cmpgt_epi16(l1_d_xor, l2_d_xor);

        // Do the 3rd quarter
        let l1_e_xor = _mm256_xor_si256(l1.e, sign);
        let l1_f_xor = _mm256_xor_si256(l1.f, sign);
        let l2_e_xor = _mm256_xor_si256(l2.e, sign);
        let l2_f_xor = _mm256_xor_si256(l2.f, sign);
        let mask_e = _mm256_cmpgt_epi16(l1_e_xor, l2_e_xor);
        let mask_f = _mm256_cmpgt_epi16(l1_f_xor, l2_f_xor);

        // Do the 4th quarter
        let l1_g_xor = _mm256_xor_si256(l1.g, sign);
        let l1_h_xor = _mm256_xor_si256(l1.h, sign);
        let l2_g_xor = _mm256_xor_si256(l2.g, sign);
        let l2_h_xor = _mm256_xor_si256(l2.h, sign);
        let mask_g = _mm256_cmpgt_epi16(l1_g_xor, l2_g_xor);
        let mask_h = _mm256_cmpgt_epi16(l1_h_xor, l2_h_xor);

        let mask = DenseLane {
            a: mask_a,
            b: mask_b,
            c: mask_c,
            d: mask_d,
            e: mask_e,
            f: mask_f,
            g: mask_g,
            h: mask_h,
        };

        apply_dense!(_mm256_srli_epi16::<15>, mask)
    }

    #[inline(always)]
    unsafe fn gte_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let maxed = apply_dense!(_mm256_max_epu16, l1, l2);
        let mask = apply_dense!(_mm256_cmpeq_epi16, l1, maxed);
        apply_dense!(_mm256_srli_epi16::<15>, mask)
    }

    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> u16 {
        // There is a bit of an assumption the compile will optimize these scalar impls
        // out, but the SIMD version is a bit complicated and is difficult to get to
        // mirror the scalar behaviour.
        // TODO: We can probably do this with the pure SIMD way, but we should try match
        //       the scalar behaviour in terms of wrapping.

        let hi = _mm256_extracti128_si256::<1>(reg);
        let lo = _mm256_castsi256_si128(reg);

        let sum = _mm_add_epi16(hi, lo);
        let unpacked = mem::transmute::<_, [u16; 8]>(sum);

        let mut s1: u16 = 0;
        let mut s2: u16 = 0;
        let mut s3: u16 = 0;
        let mut s4: u16 = 0;

        let mut i = 0;
        while i < 8 {
            s1 = s1.wrapping_add(unpacked[i]);
            s2 = s2.wrapping_add(unpacked[i + 1]);
            s3 = s3.wrapping_add(unpacked[i + 2]);
            s4 = s4.wrapping_add(unpacked[i + 3]);

            i += 4;
        }

        s1 = s1.wrapping_add(s2);
        s3 = s3.wrapping_add(s4);

        s1.wrapping_add(s3)
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> u16 {
        let hi = _mm256_extracti128_si256::<1>(reg);
        let lo = _mm256_castsi256_si128(reg);

        let maxed = _mm_max_epu16(hi, lo);
        let unpacked = mem::transmute::<_, [u16; 8]>(maxed);

        let mut m1 = u16::MIN;
        let mut m2 = u16::MIN;
        let mut m3 = u16::MIN;
        let mut m4 = u16::MIN;

        let mut i = 0;
        while i < 8 {
            m1 = m1.max(unpacked[i]);
            m2 = m2.max(unpacked[i + 1]);
            m3 = m3.max(unpacked[i + 2]);
            m4 = m4.max(unpacked[i + 3]);

            i += 4;
        }

        m1 = m1.max(m2);
        m3 = m3.max(m4);

        m1.max(m3)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> u16 {
        let hi = _mm256_extracti128_si256::<1>(reg);
        let lo = _mm256_castsi256_si128(reg);

        let minimal = _mm_min_epu16(hi, lo);
        let unpacked = mem::transmute::<_, [u16; 8]>(minimal);

        let mut m1 = u16::MAX;
        let mut m2 = u16::MAX;
        let mut m3 = u16::MAX;
        let mut m4 = u16::MAX;

        let mut i = 0;
        while i < 8 {
            m1 = m1.min(unpacked[i]);
            m2 = m2.min(unpacked[i + 1]);
            m3 = m3.min(unpacked[i + 2]);
            m4 = m4.min(unpacked[i + 3]);

            i += 4;
        }

        m1 = m1.min(m2);
        m3 = m3.min(m4);

        m1.min(m3)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut u16, reg: Self::Register) {
        _mm256_storeu_si256(mem.cast(), reg)
    }
}

impl SimdRegister<u32> for Avx2 {
    type Register = __m256i;

    #[inline(always)]
    unsafe fn load(mem: *const u32) -> Self::Register {
        _mm256_loadu_si256(mem.cast())
    }

    #[inline(always)]
    unsafe fn filled(value: u32) -> Self::Register {
        _mm256_set1_epi32(value as i32)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        _mm256_setzero_si256()
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_add_epi32(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_sub_epi32(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_mullo_epi32(l1, l2)
    }

    #[inline(always)]
    /// Scalar `u32` integer division.
    ///
    /// In reality this operation is not SIMD, in theory we could later support
    /// it however it will always be an incredibly expensive operation with quite
    /// a lot of cognitive load on the maintenance side, so for the foreseeable future
    /// non-floating point division operations will be non-simd.
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let l1_unpacked = mem::transmute::<_, [u32; 8]>(l1);
        let l2_unpacked = mem::transmute::<_, [u32; 8]>(l2);

        let mut result = [0u32; 8];
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
        let res = <Self as SimdRegister<u32>>::mul(l1, l2);
        <Self as SimdRegister<u32>>::add(res, acc)
    }

    #[inline(always)]
    unsafe fn max(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_max_epu32(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_min_epu32(l1, l2)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Self as SimdRegister<i32>>::eq(l1, l2) // Operation is identical
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Self as SimdRegister<i32>>::neq(l1, l2) // Operation is identical
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Self as SimdRegister<u32>>::gt(l2, l1)
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Self as SimdRegister<u32>>::gte(l2, l1)
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let sign = _mm256_set1_epi32(0x80000000u32 as i32);
        let l1_xor = _mm256_xor_si256(l1, sign);
        let l2_xor = _mm256_xor_si256(l2, sign);
        let mask = _mm256_cmpgt_epi32(l1_xor, l2_xor);
        // Small optimization for 16, 32 and 64bit values which
        // can shift instead of doing a bitwise `and` on a mask
        _mm256_srli_epi32::<31>(mask)
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let mask = _mm256_cmpeq_epi32(l1, _mm256_max_epu32(l1, l2));
        // Small optimization for 16, 32 and 64bit values which
        // can shift instead of doing a bitwise `and` on a mask
        _mm256_srli_epi32::<31>(mask)
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
    unsafe fn eq_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        <Self as SimdRegister<i32>>::eq_dense(l1, l2)
    }

    #[inline(always)]
    unsafe fn neq_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        <Self as SimdRegister<i32>>::neq_dense(l1, l2)
    }

    #[inline(always)]
    unsafe fn lt_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        <Self as SimdRegister<u32>>::gt_dense(l2, l1)
    }

    #[inline(always)]
    unsafe fn lte_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        <Self as SimdRegister<u32>>::gte_dense(l2, l1)
    }

    #[inline(always)]
    unsafe fn gt_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let sign = _mm256_set1_epi32(0x80000000u32 as i32);

        // We have to split the dense lane operations up into specific parts
        // because otherwise we run out of registers. Here we target no more than 16
        // registers in use.
        //
        // Do the 1st quarter of the dense lane
        let l1_a_xor = _mm256_xor_si256(l1.a, sign);
        let l1_b_xor = _mm256_xor_si256(l1.b, sign);
        let l2_a_xor = _mm256_xor_si256(l2.a, sign);
        let l2_b_xor = _mm256_xor_si256(l2.b, sign);
        let mask_a = _mm256_cmpgt_epi32(l1_a_xor, l2_a_xor);
        let mask_b = _mm256_cmpgt_epi32(l1_b_xor, l2_b_xor);

        // Do the 2nd quarter of the dense lane
        let l2_c_xor = _mm256_xor_si256(l2.c, sign);
        let l2_d_xor = _mm256_xor_si256(l2.d, sign);
        let l1_c_xor = _mm256_xor_si256(l1.c, sign);
        let l1_d_xor = _mm256_xor_si256(l1.d, sign);
        let mask_c = _mm256_cmpgt_epi32(l1_c_xor, l2_c_xor);
        let mask_d = _mm256_cmpgt_epi32(l1_d_xor, l2_d_xor);

        // Do the 3rd quarter
        let l1_e_xor = _mm256_xor_si256(l1.e, sign);
        let l1_f_xor = _mm256_xor_si256(l1.f, sign);
        let l2_e_xor = _mm256_xor_si256(l2.e, sign);
        let l2_f_xor = _mm256_xor_si256(l2.f, sign);
        let mask_e = _mm256_cmpgt_epi32(l1_e_xor, l2_e_xor);
        let mask_f = _mm256_cmpgt_epi32(l1_f_xor, l2_f_xor);

        // Do the 4th quarter
        let l1_g_xor = _mm256_xor_si256(l1.g, sign);
        let l1_h_xor = _mm256_xor_si256(l1.h, sign);
        let l2_g_xor = _mm256_xor_si256(l2.g, sign);
        let l2_h_xor = _mm256_xor_si256(l2.h, sign);
        let mask_g = _mm256_cmpgt_epi32(l1_g_xor, l2_g_xor);
        let mask_h = _mm256_cmpgt_epi32(l1_h_xor, l2_h_xor);

        let mask = DenseLane {
            a: mask_a,
            b: mask_b,
            c: mask_c,
            d: mask_d,
            e: mask_e,
            f: mask_f,
            g: mask_g,
            h: mask_h,
        };

        apply_dense!(_mm256_srli_epi32::<31>, mask)
    }

    #[inline(always)]
    unsafe fn gte_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let maxed = apply_dense!(_mm256_max_epu32, l1, l2);
        let mask = apply_dense!(_mm256_cmpeq_epi32, l1, maxed);
        apply_dense!(_mm256_srli_epi32::<31>, mask)
    }

    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> u32 {
        // There is a bit of an assumption the compile will optimize these scalar impls
        // out, but the SIMD version is a bit complicated and is difficult to get to
        // mirror the scalar behaviour.
        // TODO: We can probably do this with the pure SIMD way, but we should try match
        //       the scalar behaviour in terms of wrapping.

        let hi = _mm256_extracti128_si256::<1>(reg);
        let lo = _mm256_castsi256_si128(reg);

        let sum = _mm_add_epi32(hi, lo);
        let unpacked = mem::transmute::<__m128i, [u32; 4]>(sum);

        let mut s1 = unpacked[0];
        let s2 = unpacked[1];
        let mut s3 = unpacked[2];
        let s4 = unpacked[3];

        s1 = s1.wrapping_add(s2);
        s3 = s3.wrapping_add(s4);

        s1.wrapping_add(s3)
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> u32 {
        let hi = _mm256_extracti128_si256::<1>(reg);
        let lo = _mm256_castsi256_si128(reg);

        let maxed = _mm_max_epu32(hi, lo);
        let unpacked = mem::transmute::<__m128i, [u32; 4]>(maxed);

        let mut m1 = unpacked[0];
        let m2 = unpacked[1];
        let mut m3 = unpacked[2];
        let m4 = unpacked[3];

        m1 = m1.max(m2);
        m3 = m3.max(m4);

        m1.max(m3)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> u32 {
        let hi = _mm256_extracti128_si256::<1>(reg);
        let lo = _mm256_castsi256_si128(reg);

        let minimal = _mm_min_epu32(hi, lo);
        let unpacked = mem::transmute::<__m128i, [u32; 4]>(minimal);

        let mut m1 = unpacked[0];
        let m2 = unpacked[1];
        let mut m3 = unpacked[2];
        let m4 = unpacked[3];

        m1 = m1.min(m2);
        m3 = m3.min(m4);

        m1.min(m3)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut u32, reg: Self::Register) {
        _mm256_storeu_si256(mem.cast(), reg)
    }
}

impl SimdRegister<u64> for Avx2 {
    type Register = __m256i;

    #[inline(always)]
    unsafe fn load(mem: *const u64) -> Self::Register {
        _mm256_loadu_si256(mem.cast())
    }

    #[inline(always)]
    unsafe fn filled(value: u64) -> Self::Register {
        _mm256_set1_epi64x(value as i64)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        _mm256_setzero_si256()
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_add_epi64(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        _mm256_sub_epi64(l1, l2)
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
        let l1_unpacked = mem::transmute::<Self::Register, [u64; 4]>(l1);
        let l2_unpacked = mem::transmute::<Self::Register, [u64; 4]>(l2);

        let mut result = [0u64; 4];
        for (idx, (l1, l2)) in zip(l1_unpacked, l2_unpacked).enumerate() {
            result[idx] = l1.wrapping_div(l2);
        }

        mem::transmute::<[u64; 4], Self::Register>(result)
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
        let sign_bit = _mm256_set1_epi64x(0x8000_0000_0000_0000u64 as i64);
        let mask = _mm256_cmpgt_epi64(
            _mm256_xor_si256(l1, sign_bit),
            _mm256_xor_si256(l2, sign_bit),
        );
        _mm256_blendv_epi8(l2, l1, mask)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let sign_bit = _mm256_set1_epi64x(0x8000_0000_0000_0000u64 as i64);
        let mask = _mm256_cmpgt_epi64(
            _mm256_xor_si256(l1, sign_bit),
            _mm256_xor_si256(l2, sign_bit),
        );
        _mm256_blendv_epi8(l1, l2, mask)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Self as SimdRegister<i64>>::eq(l1, l2) // Operation is identical
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Self as SimdRegister<i64>>::neq(l1, l2) // Operation is identical
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Self as SimdRegister<u64>>::gt(l2, l1)
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        <Self as SimdRegister<u64>>::gte(l2, l1)
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        let sign = _mm256_set1_epi64x(0x8000000000000000u64 as i64);
        let l1_xor = _mm256_xor_si256(l1, sign);
        let l2_xor = _mm256_xor_si256(l2, sign);
        let mask = _mm256_cmpgt_epi64(l1_xor, l2_xor);
        // Small optimization for 16, 32 and 64bit values which
        // can shift instead of doing a bitwise `and` on a mask
        _mm256_srli_epi64::<63>(mask)
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        // A mirror of the `gt` impl but with l1 and l2 swapped operationally.
        // we can't call `Self::gt` here because we choose to shift the mask so it becomes
        // `0` and `1`s instead of the full mask.
        let swapped_cmp = {
            let sign = _mm256_set1_epi64x(0x8000000000000000u64 as i64);
            let l1_xor = _mm256_xor_si256(l1, sign);
            let l2_xor = _mm256_xor_si256(l2, sign);
            _mm256_cmpgt_epi64(l2_xor, l1_xor) // This is the important bit :)
        };

        // Because we have to do a bitwise not using a broadcast value, we can
        // cheat and just use andnot as a fused operation for also converting our mask
        _mm256_andnot_si256(swapped_cmp, _mm256_set1_epi64x(1))
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
    unsafe fn eq_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        <Self as SimdRegister<i64>>::eq_dense(l1, l2)
    }

    #[inline(always)]
    unsafe fn neq_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        <Self as SimdRegister<i64>>::neq_dense(l1, l2)
    }

    #[inline(always)]
    unsafe fn lt_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        <Self as SimdRegister<u64>>::gt_dense(l2, l1)
    }

    #[inline(always)]
    unsafe fn lte_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        <Self as SimdRegister<u64>>::gte_dense(l2, l1)
    }

    #[inline(always)]
    unsafe fn gt_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let sign = _mm256_set1_epi64x(0x8000000000000000u64 as i64);

        // We have to split the dense lane operations up into specific parts
        // because otherwise we run out of registers. Here we target no more than 16
        // registers in use.
        //
        // Do the 1st quarter of the dense lane
        let l1_a_xor = _mm256_xor_si256(l1.a, sign);
        let l1_b_xor = _mm256_xor_si256(l1.b, sign);
        let l2_a_xor = _mm256_xor_si256(l2.a, sign);
        let l2_b_xor = _mm256_xor_si256(l2.b, sign);
        let mask_a = _mm256_cmpgt_epi64(l1_a_xor, l2_a_xor);
        let mask_b = _mm256_cmpgt_epi64(l1_b_xor, l2_b_xor);

        // Do the 2nd quarter of the dense lane
        let l2_c_xor = _mm256_xor_si256(l2.c, sign);
        let l2_d_xor = _mm256_xor_si256(l2.d, sign);
        let l1_c_xor = _mm256_xor_si256(l1.c, sign);
        let l1_d_xor = _mm256_xor_si256(l1.d, sign);
        let mask_c = _mm256_cmpgt_epi64(l1_c_xor, l2_c_xor);
        let mask_d = _mm256_cmpgt_epi64(l1_d_xor, l2_d_xor);

        // Do the 3rd quarter
        let l1_e_xor = _mm256_xor_si256(l1.e, sign);
        let l1_f_xor = _mm256_xor_si256(l1.f, sign);
        let l2_e_xor = _mm256_xor_si256(l2.e, sign);
        let l2_f_xor = _mm256_xor_si256(l2.f, sign);
        let mask_e = _mm256_cmpgt_epi64(l1_e_xor, l2_e_xor);
        let mask_f = _mm256_cmpgt_epi64(l1_f_xor, l2_f_xor);

        // Do the 4th quarter
        let l1_g_xor = _mm256_xor_si256(l1.g, sign);
        let l1_h_xor = _mm256_xor_si256(l1.h, sign);
        let l2_g_xor = _mm256_xor_si256(l2.g, sign);
        let l2_h_xor = _mm256_xor_si256(l2.h, sign);
        let mask_g = _mm256_cmpgt_epi64(l1_g_xor, l2_g_xor);
        let mask_h = _mm256_cmpgt_epi64(l1_h_xor, l2_h_xor);

        let mask = DenseLane {
            a: mask_a,
            b: mask_b,
            c: mask_c,
            d: mask_d,
            e: mask_e,
            f: mask_f,
            g: mask_g,
            h: mask_h,
        };

        apply_dense!(_mm256_srli_epi64::<63>, mask)
    }

    #[inline(always)]
    unsafe fn gte_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        // A mirror of the `gt` impl but with l1 and l2 swapped operationally.
        // we can't call `Self::gt_dense` here because we choose to shift the mask so it becomes
        // `0` and `1`s instead of the full mask.
        let swapped_cmp = {
            let sign = _mm256_set1_epi64x(0x8000000000000000u64 as i64);

            // We have to split the dense lane operations up into specific parts
            // because otherwise we run out of registers. Here we target no more than 16
            // registers in use.
            //
            // Do the 1st quarter of the dense lane
            let l1_a_xor = _mm256_xor_si256(l1.a, sign);
            let l1_b_xor = _mm256_xor_si256(l1.b, sign);
            let l2_a_xor = _mm256_xor_si256(l2.a, sign);
            let l2_b_xor = _mm256_xor_si256(l2.b, sign);
            let mask_a = _mm256_cmpgt_epi64(l2_a_xor, l1_a_xor);
            let mask_b = _mm256_cmpgt_epi64(l2_b_xor, l1_b_xor);

            // Do the 2nd quarter of the dense lane
            let l2_c_xor = _mm256_xor_si256(l2.c, sign);
            let l2_d_xor = _mm256_xor_si256(l2.d, sign);
            let l1_c_xor = _mm256_xor_si256(l1.c, sign);
            let l1_d_xor = _mm256_xor_si256(l1.d, sign);
            let mask_c = _mm256_cmpgt_epi64(l2_c_xor, l1_c_xor);
            let mask_d = _mm256_cmpgt_epi64(l2_d_xor, l1_d_xor);

            // Do the 3rd quarter
            let l1_e_xor = _mm256_xor_si256(l1.e, sign);
            let l1_f_xor = _mm256_xor_si256(l1.f, sign);
            let l2_e_xor = _mm256_xor_si256(l2.e, sign);
            let l2_f_xor = _mm256_xor_si256(l2.f, sign);
            let mask_e = _mm256_cmpgt_epi64(l2_e_xor, l1_e_xor);
            let mask_f = _mm256_cmpgt_epi64(l2_f_xor, l1_f_xor);

            // Do the 4th quarter
            let l1_g_xor = _mm256_xor_si256(l1.g, sign);
            let l1_h_xor = _mm256_xor_si256(l1.h, sign);
            let l2_g_xor = _mm256_xor_si256(l2.g, sign);
            let l2_h_xor = _mm256_xor_si256(l2.h, sign);
            let mask_g = _mm256_cmpgt_epi64(l2_g_xor, l1_g_xor);
            let mask_h = _mm256_cmpgt_epi64(l2_h_xor, l1_h_xor);

            DenseLane {
                a: mask_a,
                b: mask_b,
                c: mask_c,
                d: mask_d,
                e: mask_e,
                f: mask_f,
                g: mask_g,
                h: mask_h,
            }
        };

        apply_dense!(
            _mm256_andnot_si256,
            swapped_cmp,
            value = _mm256_set1_epi64x(1)
        )
    }

    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> u64 {
        // There is a bit of an assumption the compile will optimize these scalar impls
        // out, but the SIMD version is a bit complicated and is difficult to get to
        // mirror the scalar behaviour.
        // TODO: We can probably do this with the pure SIMD way, but we should try match
        //       the scalar behaviour in terms of wrapping.

        let hi = _mm256_extracti128_si256::<1>(reg);
        let lo = _mm256_castsi256_si128(reg);

        let sum = _mm_add_epi64(hi, lo);
        let unpacked = mem::transmute::<__m128i, [u64; 2]>(sum);

        let s1 = unpacked[0];
        let s2 = unpacked[1];

        s1.wrapping_add(s2)
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> u64 {
        let hi = _mm256_extracti128_si256::<1>(reg);
        let lo = _mm256_castsi256_si128(reg);

        let sign_bit = _mm_set1_epi64x(0x8000_0000_0000_0000u64 as i64);
        let mask =
            _mm_cmpgt_epi64(_mm_xor_si128(hi, sign_bit), _mm_xor_si128(lo, sign_bit));
        let max = _mm_blendv_epi8(lo, hi, mask);

        let [m1, m2] = mem::transmute::<__m128i, [u64; 2]>(max);

        m1.max(m2)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> u64 {
        let hi = _mm256_extracti128_si256::<1>(reg);
        let lo = _mm256_castsi256_si128(reg);

        let sign_bit = _mm_set1_epi64x(0x8000_0000_0000_0000u64 as i64);
        let mask =
            _mm_cmpgt_epi64(_mm_xor_si128(hi, sign_bit), _mm_xor_si128(lo, sign_bit));
        let min = _mm_blendv_epi8(hi, lo, mask);

        let [m1, m2] = mem::transmute::<__m128i, [u64; 2]>(min);

        m1.min(m2)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut u64, reg: Self::Register) {
        _mm256_storeu_si256(mem.cast(), reg)
    }
}

impl Hypot<f64> for Avx2 {
    #[inline(always)]
    unsafe fn hypot(x: Self::Register, y: Self::Register) -> Self::Register {
        // convert the inputs to absolute values

        let x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
        let y = _mm256_andnot_pd(_mm256_set1_pd(-0.0), y);
        let hi = <Avx2 as SimdRegister<f64>>::max(x, y);
        let lo = <Avx2 as SimdRegister<f64>>::min(x, y);
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
        let lo_scaled = _mm256_mul_pd(lo_scaled, lo_scaled);
        _mm256_mul_pd(hi2p, _mm256_sqrt_pd(_mm256_add_pd(lo_scaled, hi_scaled)))
    }
}
