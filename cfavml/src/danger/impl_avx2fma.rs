#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::core_simd_api::SimdRegister;
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
