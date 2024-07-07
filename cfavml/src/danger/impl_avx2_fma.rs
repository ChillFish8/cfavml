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

#[cfg(all(target_feature = "avx2", target_feature = "fma", test))]
mod tests {
    use super::*;
    use crate::test_utils::get_sample_vectors;

    #[test]
    fn test_suite() {
        unsafe { crate::danger::impl_test::test_suite_impl_f32::<Avx2Fma>() }
    }

    #[test]
    fn test_cosine() {
        let (l1, l2) = get_sample_vectors::<f32>(1043);
        unsafe { crate::danger::op_cosine::test_cosine::<_, Avx2Fma>(l1, l2) };

        let (l1, l2) = get_sample_vectors::<f64>(1043);
        unsafe { crate::danger::op_cosine::test_cosine::<_, Avx2Fma>(l1, l2) };
    }

    #[test]
    fn test_dot_product() {
        let (l1, l2) = get_sample_vectors::<f32>(1043);
        unsafe { crate::danger::op_dot_product::test_dot::<_, Avx2Fma>(l1, l2) };

        let (l1, l2) = get_sample_vectors::<f64>(1043);
        unsafe { crate::danger::op_dot_product::test_dot::<_, Avx2Fma>(l1, l2) };
    }

    #[test]
    fn test_norm() {
        let (l1, _) = get_sample_vectors::<f32>(1043);
        unsafe { crate::danger::op_norm::test_norm::<_, Avx2Fma>(l1) };

        let (l1, _) = get_sample_vectors::<f64>(1043);
        unsafe { crate::danger::op_norm::test_norm::<_, Avx2Fma>(l1) };
    }

    #[test]
    fn test_euclidean() {
        let (l1, l2) = get_sample_vectors::<f32>(1043);
        unsafe { crate::danger::op_euclidean::test_euclidean::<_, Avx2Fma>(l1, l2) };

        let (l1, l2) = get_sample_vectors::<f64>(1043);
        unsafe { crate::danger::op_euclidean::test_euclidean::<_, Avx2Fma>(l1, l2) };
    }

    #[test]
    fn test_max() {
        let (l1, l2) = get_sample_vectors::<f32>(1043);
        unsafe { crate::danger::op_max::test_max::<_, Avx2Fma>(l1, l2) };

        let (l1, l2) = get_sample_vectors::<f64>(1043);
        unsafe { crate::danger::op_max::test_max::<_, Avx2Fma>(l1, l2) };
    }

    #[test]
    fn test_min() {
        let (l1, l2) = get_sample_vectors::<f32>(1043);
        unsafe { crate::danger::op_min::test_min::<_, Avx2Fma>(l1, l2) };

        let (l1, l2) = get_sample_vectors::<f64>(1043);
        unsafe { crate::danger::op_min::test_min::<_, Avx2Fma>(l1, l2) };
    }

    #[test]
    fn test_sum() {
        let (l1, l2) = (vec![1.0f32; 1043], vec![3.0f32; 1043]);
        unsafe { crate::danger::op_sum::test_sum::<_, Avx2Fma>(l1, l2) };

        let (l1, l2) = (vec![1.0f32; 1043], vec![3.0f32; 1043]);
        unsafe { crate::danger::op_sum::test_sum::<_, Avx2Fma>(l1, l2) };
    }

    #[test]
    fn test_vector_x_value() {
        let (l1, _) = (vec![1.0f32; 1043], vec![3.0f32; 1043]);
        unsafe {
            crate::danger::op_vector_x_value::tests::test_add::<_, Avx2Fma>(
                l1.clone(),
                2.0,
            )
        };
        unsafe {
            crate::danger::op_vector_x_value::tests::test_sub::<_, Avx2Fma>(
                l1.clone(),
                2.0,
            )
        };
        unsafe {
            crate::danger::op_vector_x_value::tests::test_div::<_, Avx2Fma>(
                l1.clone(),
                2.0,
            )
        };
        unsafe {
            crate::danger::op_vector_x_value::tests::test_mul::<_, Avx2Fma>(l1, 2.0)
        };
    }

    #[test]
    fn test_vector_x_vector() {
        let (l1, l2) = (vec![1.0f32; 1043], vec![3.0f32; 1043]);
        unsafe {
            crate::danger::op_vector_x_vector::tests::test_add::<_, Avx2Fma>(
                l1.clone(),
                l2.clone(),
            )
        };
        unsafe {
            crate::danger::op_vector_x_vector::tests::test_sub::<_, Avx2Fma>(
                l1.clone(),
                l2.clone(),
            )
        };
        unsafe {
            crate::danger::op_vector_x_vector::tests::test_div::<_, Avx2Fma>(
                l1.clone(),
                l2.clone(),
            )
        };
        unsafe {
            crate::danger::op_vector_x_vector::tests::test_mul::<_, Avx2Fma>(l1, l2)
        };
    }
}
