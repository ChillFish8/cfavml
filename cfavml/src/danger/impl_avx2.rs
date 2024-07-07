use core::arch::x86_64::*;
use core::iter::zip;
use core::mem;

use super::core_simd_api::{DenseLane, SimdRegister};
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
        let [a, b, c, d] = mem::transmute::<_, [f32; 4]>(maxed);

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
    unsafe fn fmadd_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
        acc: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        let mask = DenseLane::copy(_mm256_set1_epi32(0xFF00FF00u32 as i32));

        let even = apply_dense!(_mm256_mullo_epi16, l1, l2);

        let shift_l1 = apply_dense!(_mm256_srai_epi16::<8>, l1);
        let shift_l2 = apply_dense!(_mm256_srai_epi16::<8>, l2);

        let odd = apply_dense!(_mm256_mullo_epi16, shift_l1, shift_l2);
        let odd = apply_dense!(_mm256_slli_epi16::<8>, odd);

        let res = apply_dense!(_mm256_blendv_epi8, even, odd, mask);
        apply_dense!(_mm256_add_epi8, res, acc)
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

        let mut s1 = unpacked[0].wrapping_add(unpacked[1]);
        let s2 = unpacked[2].wrapping_add(unpacked[3]);
        let mut s3 = unpacked[4].wrapping_add(unpacked[5]);
        let s4 = unpacked[6].wrapping_add(unpacked[7]);
        let mut s5 = unpacked[8].wrapping_add(unpacked[9]);
        let s6 = unpacked[10].wrapping_add(unpacked[11]);
        let mut s7 = unpacked[12].wrapping_add(unpacked[13]);
        let s8 = unpacked[14].wrapping_add(unpacked[15]);

        s1 = s1.wrapping_add(s2);
        s3 = s3.wrapping_add(s4);
        s5 = s5.wrapping_add(s6);
        s7 = s7.wrapping_add(s8);

        s1 = s1.wrapping_add(s3);
        s5 = s5.wrapping_add(s7);

        s5.wrapping_add(s1)
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
