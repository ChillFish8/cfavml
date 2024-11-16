use core::arch::aarch64::*;
use core::iter::zip;
use core::mem;

use super::core_simd_api::{DenseLane, SimdRegister,Hypot};
use crate::math::{AutoMath, Math};

const BITS_8_CAPACITY: usize = 16;
const BITS_16_CAPACITY: usize = 8;
const BITS_32_CAPACITY: usize = 4;
const BITS_64_CAPACITY: usize = 2;

/// NEON enabled SIMD operations.
///
/// This requires the `neon` CPU features be enabled.
pub struct Neon;

impl SimdRegister<f32> for Neon {
    type Register = float32x4_t;

    #[inline(always)]
    unsafe fn load(mem: *const f32) -> Self::Register {
        vld1q_f32(mem)
    }

    #[inline(always)]
    unsafe fn filled(value: f32) -> Self::Register {
        vdupq_n_f32(value)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        <Self as SimdRegister<f32>>::filled(0.0)
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vaddq_f32(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vsubq_f32(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vmulq_f32(l1, l2)
    }

    #[inline(always)]
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vdivq_f32(l1, l2)
    }

    #[inline(always)]
    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        vfmaq_f32(acc, l1, l2)
    }

    #[inline(always)]
    unsafe fn max(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vmaxq_f32(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vminq_f32(l1, l2)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<f32, Self::Register, _, BITS_32_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_eq(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<f32, Self::Register, _, BITS_32_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(!AutoMath::cmp_eq(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<f32, Self::Register, _, BITS_32_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_lt(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<f32, Self::Register, _, BITS_32_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_lte(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<f32, Self::Register, _, BITS_32_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_gt(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<f32, Self::Register, _, BITS_32_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_gte(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> f32 {
        vaddvq_f32(reg)
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> f32 {
        vmaxvq_f32(reg)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> f32 {
        vminvq_f32(reg)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut f32, reg: Self::Register) {
        vst1q_f32(mem, reg)
    }
}
const EXPONENT_MASK_F32: u32 = 2139095040;
const MANTISSA_MASK_F32: u32 = 8388607;
impl Hypot<f32> for Neon {
    #[inline(always)]
    unsafe fn hypot(x: Self::Register, y: Self::Register) -> Self::Register {
        // Convert inputs to absolute values
        let (x, y) = (vabsq_f32(x), vabsq_f32(y));

        // Find the max and min of the two inputs
        let (hi, lo) = (vmaxq_f32(x, y), vminq_f32(x, y));
        let exponent_mask = vdupq_n_u32(EXPONENT_MASK_F32);
        let mantissa_mask = vdupq_n_u32(MANTISSA_MASK_F32);

        // round the hi values down to the nearest power of 2
        let hi2p =
            vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(hi), exponent_mask));
        // we scale the values inside the root by the reciprocal of hi2p. since it's a power of 2,
        // we can double it and xor it with the exponent mask
        let scale = vreinterpretq_f32_u32(veorq_u32(
            vreinterpretq_u32_f32(vaddq_f32(hi2p, hi2p)),
            exponent_mask,
        ));
        // create a mask that matches the normal hi values
        let mask = vcgtq_f32(hi, vdupq_n_f32(f32::MIN_POSITIVE));
        // replace the subnormal values of hi2p with the minimum positive normal value
        let hi2p = vbslq_f32(mask, hi2p, vdupq_n_f32(f32::MIN_POSITIVE));
        // replace the subnormal values of scale with the reciprocal of the minimum positive normal value
        let scale = vbslq_f32(mask, scale, vdupq_n_f32(1.0 / f32::MIN_POSITIVE));
        // create a mask that matches the subnormal hi values
        let mask = vcltq_f32(hi, vdupq_n_f32(f32::MIN_POSITIVE));
        // since hi2p was preserved the exponent bits of hi, the exponent of hi/hi2p is 1
        let hi_scaled = vreinterpretq_f32_u32(vorrq_u32(
            vandq_u32(vreinterpretq_u32_f32(hi), mantissa_mask),
            vreinterpretq_u32_f32(vdupq_n_f32(1.0)),
        ));
        // for the subnormal elements of hi, we need to subtract 1 from the scaled hi values
        let hi_scaled =
            vbslq_f32(mask, vsubq_f32(hi_scaled, vdupq_n_f32(1.0)), hi_scaled);
        // finally, do the thing
        let hi_scaled = vmulq_f32(hi_scaled, hi_scaled);
        let lo_scaled = vmulq_f32(lo, scale);
        vmulq_f32(hi2p, vsqrtq_f32(vfmaq_f32(lo_scaled, lo_scaled, hi_scaled)))
    }
}

impl SimdRegister<f64> for Neon {
    type Register = float64x2_t;

    #[inline(always)]
    unsafe fn load(mem: *const f64) -> Self::Register {
        vld1q_f64(mem)
    }

    #[inline(always)]
    unsafe fn filled(value: f64) -> Self::Register {
        vdupq_n_f64(value)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        <Self as SimdRegister<f64>>::filled(0.0)
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vaddq_f64(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vsubq_f64(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vmulq_f64(l1, l2)
    }

    #[inline(always)]
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vdivq_f64(l1, l2)
    }

    #[inline(always)]
    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        vfmaq_f64(acc, l1, l2)
    }

    #[inline(always)]
    unsafe fn max(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vmaxq_f64(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vminq_f64(l1, l2)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<f64, Self::Register, _, BITS_64_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_eq(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<f64, Self::Register, _, BITS_64_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(!AutoMath::cmp_eq(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<f64, Self::Register, _, BITS_64_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_lt(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<f64, Self::Register, _, BITS_64_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_lte(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<f64, Self::Register, _, BITS_64_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_gt(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<f64, Self::Register, _, BITS_64_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_gte(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> f64 {
        vaddvq_f64(reg)
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> f64 {
        vmaxvq_f64(reg)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> f64 {
        vminvq_f64(reg)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut f64, reg: Self::Register) {
        vst1q_f64(mem, reg)
    }
}

impl Hypot<f64> for Neon {
    #[inline(always)]
    unsafe fn hypot(x: Self::Register, y: Self::Register) -> Self::Register {
        // Convert inputs to absolute values
        let (x, y) = (vabsq_f64(x), vabsq_f64(y));

        // Find the max and min of the two inputs
        let (hi, lo) = (vmaxq_f64(x, y), vminq_f64(x, y));
        let exponent_mask = vdupq_n_u64(f64::INFINITY.to_bits());
        let mantissa_mask = vdupq_n_u64((f64::MIN_POSITIVE - mem::transmute::<u64,f64>(1)).to_bits());

        // round the hi values down to the nearest power of 2
        let hi2p =
            vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(hi), exponent_mask));
        // we scale the values inside the root by the reciprocal of hi2p. since it's a power of 2,
        // we can double it and xor it with the exponent mask
        let scale = vreinterpretq_f64_u64(veorq_u64(
            vreinterpretq_u64_f64(vaddq_f64(hi2p, hi2p)),
            exponent_mask,
        ));
        // create a mask that matches the normal hi values
        let mask = vcgtq_f64(hi, vdupq_n_f64(f64::MIN_POSITIVE));
        // replace the subnormal values of hi2p with the minimum positive normal value
        let hi2p = vbslq_f64(mask, hi2p, vdupq_n_f64(f64::MIN_POSITIVE));
        // replace the subnormal values of scale with the reciprocal of the minimum positive normal value
        let scale = vbslq_f64(mask, scale, vdupq_n_f64(1.0 / f64::MIN_POSITIVE));
        // create a mask that matches the subnormal hi values
        let mask = vcltq_f64(hi, vdupq_n_f64(f64::MIN_POSITIVE));
        // since hi2p was preserved the exponent bits of hi, the exponent of hi/hi2p is 1
        let hi_scaled = vreinterpretq_f64_u64(vorrq_u64(
            vandq_u64(vreinterpretq_u64_f64(hi), mantissa_mask),
            vreinterpretq_u64_f64(vdupq_n_f64(1.0)),
        ));
        // for the subnormal elements of hi, we need to subtract 1 from the scaled hi values
        let hi_scaled =
            vbslq_f64(mask, vsubq_f64(hi_scaled, vdupq_n_f64(1.0)), hi_scaled);
        // finally, do the thing
        let hi_scaled = vmulq_f64(hi_scaled, hi_scaled);
        let lo_scaled = vmulq_f64(lo, scale);
        vmulq_f64(hi2p, vsqrtq_f64(vfmaq_f64(lo_scaled, lo_scaled, hi_scaled)))
    }
}

impl SimdRegister<i8> for Neon {
    type Register = int8x16_t;

    #[inline(always)]
    unsafe fn load(mem: *const i8) -> Self::Register {
        vld1q_s8(mem)
    }

    #[inline(always)]
    unsafe fn filled(value: i8) -> Self::Register {
        vdupq_n_s8(value)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        <Self as SimdRegister<i8>>::filled(0)
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vaddq_s8(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vsubq_s8(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vmulq_s8(l1, l2)
    }

    #[inline(always)]
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i8, Self::Register, _, BITS_8_CAPACITY>(l1, l2, |a, b| {
            AutoMath::div(a, b)
        })
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
        vmaxq_s8(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vminq_s8(l1, l2)
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
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i8, Self::Register, _, BITS_8_CAPACITY>(l1, l2, |a, b| {
            AutoMath::cast_bool(AutoMath::cmp_eq(a, b))
        })
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i8, Self::Register, _, BITS_8_CAPACITY>(l1, l2, |a, b| {
            AutoMath::cast_bool(!AutoMath::cmp_eq(a, b))
        })
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i8, Self::Register, _, BITS_8_CAPACITY>(l1, l2, |a, b| {
            AutoMath::cast_bool(AutoMath::cmp_lt(a, b))
        })
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i8, Self::Register, _, BITS_8_CAPACITY>(l1, l2, |a, b| {
            AutoMath::cast_bool(AutoMath::cmp_lte(a, b))
        })
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i8, Self::Register, _, BITS_8_CAPACITY>(l1, l2, |a, b| {
            AutoMath::cast_bool(AutoMath::cmp_gt(a, b))
        })
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i8, Self::Register, _, BITS_8_CAPACITY>(l1, l2, |a, b| {
            AutoMath::cast_bool(AutoMath::cmp_gte(a, b))
        })
    }

    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> i8 {
        vaddvq_s8(reg)
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> i8 {
        vmaxvq_s8(reg)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> i8 {
        vminvq_s8(reg)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut i8, reg: Self::Register) {
        vst1q_s8(mem, reg)
    }
}

impl SimdRegister<i16> for Neon {
    type Register = int16x8_t;

    #[inline(always)]
    unsafe fn load(mem: *const i16) -> Self::Register {
        vld1q_s16(mem)
    }

    #[inline(always)]
    unsafe fn filled(value: i16) -> Self::Register {
        vdupq_n_s16(value)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        <Self as SimdRegister<i16>>::filled(0)
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vaddq_s16(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vsubq_s16(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vmulq_s16(l1, l2)
    }

    #[inline(always)]
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i16, Self::Register, _, BITS_16_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::div(a, b),
        )
    }

    #[inline(always)]
    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        let res = <Self as SimdRegister<i16>>::mul(l1, l2);
        <Self as SimdRegister<i16>>::add(res, acc)
    }

    #[inline(always)]
    unsafe fn max(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vmaxq_s16(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vminq_s16(l1, l2)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i16, Self::Register, _, BITS_16_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_eq(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i16, Self::Register, _, BITS_16_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(!AutoMath::cmp_eq(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i16, Self::Register, _, BITS_16_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_lt(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i16, Self::Register, _, BITS_16_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_lte(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i16, Self::Register, _, BITS_16_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_gt(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i16, Self::Register, _, BITS_16_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_gte(a, b)),
        )
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
        vaddvq_s16(reg)
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> i16 {
        vmaxvq_s16(reg)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> i16 {
        vminvq_s16(reg)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut i16, reg: Self::Register) {
        vst1q_s16(mem, reg)
    }
}

impl SimdRegister<i32> for Neon {
    type Register = int32x4_t;

    #[inline(always)]
    unsafe fn load(mem: *const i32) -> Self::Register {
        vld1q_s32(mem)
    }

    #[inline(always)]
    unsafe fn filled(value: i32) -> Self::Register {
        vdupq_n_s32(value)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        <Self as SimdRegister<i32>>::filled(0)
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vaddq_s32(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vsubq_s32(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vmulq_s32(l1, l2)
    }

    #[inline(always)]
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i32, Self::Register, _, BITS_32_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::div(a, b),
        )
    }

    #[inline(always)]
    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        let res = <Self as SimdRegister<i32>>::mul(l1, l2);
        <Self as SimdRegister<i32>>::add(res, acc)
    }

    #[inline(always)]
    unsafe fn max(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vmaxq_s32(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vminq_s32(l1, l2)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i32, Self::Register, _, BITS_32_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_eq(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i32, Self::Register, _, BITS_32_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(!AutoMath::cmp_eq(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i32, Self::Register, _, BITS_32_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_lt(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i32, Self::Register, _, BITS_32_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_lte(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i32, Self::Register, _, BITS_32_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_gt(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i32, Self::Register, _, BITS_32_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_gte(a, b)),
        )
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
        vaddvq_s32(reg)
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> i32 {
        vmaxvq_s32(reg)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> i32 {
        vminvq_s32(reg)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut i32, reg: Self::Register) {
        vst1q_s32(mem, reg)
    }
}

impl SimdRegister<i64> for Neon {
    type Register = int64x2_t;

    #[inline(always)]
    unsafe fn load(mem: *const i64) -> Self::Register {
        vld1q_s64(mem)
    }

    #[inline(always)]
    unsafe fn filled(value: i64) -> Self::Register {
        vdupq_n_s64(value)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        <Self as SimdRegister<i64>>::filled(0)
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vaddq_s64(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vsubq_s64(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i64, Self::Register, _, BITS_64_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::mul(a, b),
        )
    }

    #[inline(always)]
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i64, Self::Register, _, BITS_64_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::div(a, b),
        )
    }

    #[inline(always)]
    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        let res = <Self as SimdRegister<i64>>::mul(l1, l2);
        <Self as SimdRegister<i64>>::add(res, acc)
    }

    #[inline(always)]
    unsafe fn max(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i64, Self::Register, _, BITS_64_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cmp_max(a, b),
        )
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i64, Self::Register, _, BITS_64_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cmp_min(a, b),
        )
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i64, Self::Register, _, BITS_64_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_eq(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i64, Self::Register, _, BITS_64_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(!AutoMath::cmp_eq(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i64, Self::Register, _, BITS_64_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_lt(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i64, Self::Register, _, BITS_64_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_lte(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i64, Self::Register, _, BITS_64_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_gt(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<i64, Self::Register, _, BITS_64_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_gte(a, b)),
        )
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
        vaddvq_s64(reg)
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> i64 {
        let [a, b] = mem::transmute::<_, [i64; 2]>(reg);
        core::cmp::max(a, b)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> i64 {
        let [a, b] = mem::transmute::<_, [i64; 2]>(reg);
        core::cmp::min(a, b)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut i64, reg: Self::Register) {
        vst1q_s64(mem, reg)
    }
}

impl SimdRegister<u8> for Neon {
    type Register = uint8x16_t;

    #[inline(always)]
    unsafe fn load(mem: *const u8) -> Self::Register {
        vld1q_u8(mem)
    }

    #[inline(always)]
    unsafe fn filled(value: u8) -> Self::Register {
        vdupq_n_u8(value)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        <Self as SimdRegister<u8>>::filled(0)
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vaddq_u8(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vsubq_u8(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vmulq_u8(l1, l2)
    }

    #[inline(always)]
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u8, Self::Register, _, BITS_8_CAPACITY>(l1, l2, |a, b| {
            AutoMath::div(a, b)
        })
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
        vmaxq_u8(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vminq_u8(l1, l2)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u8, Self::Register, _, BITS_8_CAPACITY>(l1, l2, |a, b| {
            AutoMath::cast_bool(AutoMath::cmp_eq(a, b))
        })
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u8, Self::Register, _, BITS_8_CAPACITY>(l1, l2, |a, b| {
            AutoMath::cast_bool(!AutoMath::cmp_eq(a, b))
        })
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u8, Self::Register, _, BITS_8_CAPACITY>(l1, l2, |a, b| {
            AutoMath::cast_bool(AutoMath::cmp_lt(a, b))
        })
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u8, Self::Register, _, BITS_8_CAPACITY>(l1, l2, |a, b| {
            AutoMath::cast_bool(AutoMath::cmp_lte(a, b))
        })
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u8, Self::Register, _, BITS_8_CAPACITY>(l1, l2, |a, b| {
            AutoMath::cast_bool(AutoMath::cmp_gt(a, b))
        })
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u8, Self::Register, _, BITS_8_CAPACITY>(l1, l2, |a, b| {
            AutoMath::cast_bool(AutoMath::cmp_gte(a, b))
        })
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
        vaddvq_u8(reg)
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> u8 {
        vmaxvq_u8(reg)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> u8 {
        vminvq_u8(reg)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut u8, reg: Self::Register) {
        vst1q_u8(mem, reg)
    }
}

impl SimdRegister<u16> for Neon {
    type Register = uint16x8_t;

    #[inline(always)]
    unsafe fn load(mem: *const u16) -> Self::Register {
        vld1q_u16(mem)
    }

    #[inline(always)]
    unsafe fn filled(value: u16) -> Self::Register {
        vdupq_n_u16(value)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        <Self as SimdRegister<u16>>::filled(0)
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vaddq_u16(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vsubq_u16(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vmulq_u16(l1, l2)
    }

    #[inline(always)]
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u16, Self::Register, _, BITS_16_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::div(a, b),
        )
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
        vmaxq_u16(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vminq_u16(l1, l2)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u16, Self::Register, _, BITS_16_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_eq(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u16, Self::Register, _, BITS_16_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(!AutoMath::cmp_eq(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u16, Self::Register, _, BITS_16_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_lt(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u16, Self::Register, _, BITS_16_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_lte(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u16, Self::Register, _, BITS_16_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_gt(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u16, Self::Register, _, BITS_16_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_gte(a, b)),
        )
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
        vaddvq_u16(reg)
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> u16 {
        vmaxvq_u16(reg)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> u16 {
        vminvq_u16(reg)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut u16, reg: Self::Register) {
        vst1q_u16(mem, reg)
    }
}

impl SimdRegister<u32> for Neon {
    type Register = uint32x4_t;

    #[inline(always)]
    unsafe fn load(mem: *const u32) -> Self::Register {
        vld1q_u32(mem)
    }

    #[inline(always)]
    unsafe fn filled(value: u32) -> Self::Register {
        vdupq_n_u32(value)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        <Self as SimdRegister<u32>>::filled(0)
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vaddq_u32(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vsubq_u32(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vmulq_u32(l1, l2)
    }

    #[inline(always)]
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u32, Self::Register, _, BITS_32_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::div(a, b),
        )
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
        vmaxq_u32(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vminq_u32(l1, l2)
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u32, Self::Register, _, BITS_32_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_eq(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u32, Self::Register, _, BITS_32_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(!AutoMath::cmp_eq(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u32, Self::Register, _, BITS_32_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_lt(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u32, Self::Register, _, BITS_32_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_lte(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u32, Self::Register, _, BITS_32_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_gt(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u32, Self::Register, _, BITS_32_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_gte(a, b)),
        )
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
        vaddvq_u32(reg)
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> u32 {
        vmaxvq_u32(reg)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> u32 {
        vminvq_u32(reg)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut u32, reg: Self::Register) {
        vst1q_u32(mem, reg)
    }
}

impl SimdRegister<u64> for Neon {
    type Register = uint64x2_t;

    #[inline(always)]
    unsafe fn load(mem: *const u64) -> Self::Register {
        vld1q_u64(mem)
    }

    #[inline(always)]
    unsafe fn filled(value: u64) -> Self::Register {
        vdupq_n_u64(value)
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        <Self as SimdRegister<u64>>::filled(0)
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vaddq_u64(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        vsubq_u64(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u64, Self::Register, _, BITS_64_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::mul(a, b),
        )
    }

    #[inline(always)]
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u64, Self::Register, _, BITS_64_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::div(a, b),
        )
    }

    #[inline(always)]
    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        let res = <Self as SimdRegister<u64>>::mul(l1, l2);
        <Self as SimdRegister<u64>>::add(res, acc)
    }

    #[inline(always)]
    unsafe fn max(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u64, Self::Register, _, BITS_64_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cmp_max(a, b),
        )
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u64, Self::Register, _, BITS_64_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cmp_min(a, b),
        )
    }

    #[inline(always)]
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u64, Self::Register, _, BITS_64_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_eq(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u64, Self::Register, _, BITS_64_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(!AutoMath::cmp_eq(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u64, Self::Register, _, BITS_64_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_lt(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u64, Self::Register, _, BITS_64_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_lte(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u64, Self::Register, _, BITS_64_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_gt(a, b)),
        )
    }

    #[inline(always)]
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register {
        apply_fallback_math::<u64, Self::Register, _, BITS_64_CAPACITY>(
            l1,
            l2,
            |a, b| AutoMath::cast_bool(AutoMath::cmp_gte(a, b)),
        )
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
        vaddvq_u64(reg)
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> u64 {
        let [a, b] = mem::transmute::<_, [u64; 2]>(reg);
        core::cmp::max(a, b)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> u64 {
        let [a, b] = mem::transmute::<_, [u64; 2]>(reg);
        core::cmp::min(a, b)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut u64, reg: Self::Register) {
        vst1q_u64(mem, reg)
    }
}

#[inline]
/// A helper function for apply fallback math operations to a register.
///
/// Note this is _horrifically_ unsafe and a glorified set of transmutes.
unsafe fn apply_fallback_math<T, R, Op, const N: usize>(a: R, b: R, op: Op) -> R
where
    T: Copy,
    R: ScalarCasting<T, N>,
    AutoMath: Math<T>,
    Op: Fn(T, T) -> T,
{
    let l1_unpacked = a.to_array();
    let l2_unpacked = b.to_array();

    let mut result = [AutoMath::zero(); N];
    for (idx, (l1, l2)) in zip(l1_unpacked, l2_unpacked).enumerate() {
        result[idx] = op(l1, l2);
    }

    R::from_array(result)
}

/// A helper trait to work around transmute limitations.
///
/// TODO: We should use this for all of the transmute ops in this file
trait ScalarCasting<T, const N: usize> {
    unsafe fn to_array(self) -> [T; N];
    unsafe fn from_array(data: [T; N]) -> Self;
}

macro_rules! casting_helper {
    ($t:ident, $bits:expr, $r:ident) => {
        impl ScalarCasting<$t, $bits> for $r {
            #[inline]
            unsafe fn from_array(data: [$t; $bits]) -> Self {
                mem::transmute(data)
            }

            #[inline]
            unsafe fn to_array(self) -> [$t; $bits] {
                mem::transmute(self)
            }
        }
    };
}

casting_helper!(f32, 4, float32x4_t);
casting_helper!(f64, 2, float64x2_t);
casting_helper!(i8, 16, int8x16_t);
casting_helper!(i16, 8, int16x8_t);
casting_helper!(i32, 4, int32x4_t);
casting_helper!(i64, 2, int64x2_t);
casting_helper!(u8, 16, uint8x16_t);
casting_helper!(u16, 8, uint16x8_t);
casting_helper!(u32, 4, uint32x4_t);
casting_helper!(u64, 2, uint64x2_t);
