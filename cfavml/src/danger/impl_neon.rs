use core::arch::aarch64::*;

use crate::danger::SimdRegister;

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
