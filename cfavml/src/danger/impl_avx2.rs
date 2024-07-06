use core::arch::x86_64::*;
use std::mem;

use super::core_simd_api::{DenseLane, SimdRegister};

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
        let [a, b, c, d, e, f, g, h] = mem::transmute::<_, [f32; 8]>(reg);

        let mut m1 = a.max(b);
        let m2 = c.max(d);
        let mut m3 = e.max(f);
        let m4 = g.max(h);

        m1 = m1.max(m2);
        m3 = m3.max(m4);

        m1.max(m3)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> f32 {
        let [a, b, c, d, e, f, g, h] = mem::transmute::<_, [f32; 8]>(reg);

        let mut m1 = a.min(b);
        let m2 = c.min(d);
        let mut m3 = e.min(f);
        let m4 = g.min(h);

        m1 = m1.min(m2);
        m3 = m3.min(m4);

        m1.min(m3)
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
        let [a, b, c, d] = mem::transmute::<_, [f64; 4]>(reg);

        let m1 = a.max(b);
        let m2 = c.max(d);

        m1.max(m2)
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> f64 {
        let [a, b, c, d] = mem::transmute::<_, [f64; 4]>(reg);

        let m1 = a.min(b);
        let m2 = c.min(d);

        m1.min(m2)
    }

    #[inline(always)]
    unsafe fn write(mem: *mut f64, reg: Self::Register) {
        _mm256_storeu_pd(mem, reg)
    }
}

#[cfg(all(target_feature = "avx2", test))]
mod tests {
    use super::*;
    use crate::test_utils::get_sample_vectors;

    #[test]
    fn test_suite() {
        unsafe { crate::danger::impl_test::test_suite_impl_f32::<Avx2>() }
    }

    #[test]
    fn test_cosine() {
        let (l1, l2) = get_sample_vectors::<f32>(1043);
        unsafe { crate::danger::op_cosine::test_cosine::<_, Avx2>(l1, l2) };

        let (l1, l2) = get_sample_vectors::<f64>(1043);
        unsafe { crate::danger::op_cosine::test_cosine::<_, Avx2>(l1, l2) };
    }

    #[test]
    fn test_dot_product() {
        let (l1, l2) = get_sample_vectors::<f32>(1043);
        unsafe { crate::danger::op_dot_product::test_dot::<_, Avx2>(l1, l2) };

        let (l1, l2) = get_sample_vectors::<f64>(1043);
        unsafe { crate::danger::op_dot_product::test_dot::<_, Avx2>(l1, l2) };
    }

    #[test]
    fn test_euclidean() {
        let (l1, l2) = get_sample_vectors::<f32>(1043);
        unsafe { crate::danger::op_euclidean::test_euclidean::<_, Avx2>(l1, l2) };

        let (l1, l2) = get_sample_vectors::<f64>(1043);
        unsafe { crate::danger::op_euclidean::test_euclidean::<_, Avx2>(l1, l2) };
    }

    #[test]
    fn test_max() {
        let (l1, l2) = get_sample_vectors::<f32>(1043);
        unsafe { crate::danger::op_max::test_max::<_, Avx2>(l1, l2) };

        let (l1, l2) = get_sample_vectors::<f64>(1043);
        unsafe { crate::danger::op_max::test_max::<_, Avx2>(l1, l2) };
    }

    #[test]
    fn test_min() {
        let (l1, l2) = get_sample_vectors::<f32>(1043);
        unsafe { crate::danger::op_min::test_min::<_, Avx2>(l1, l2) };

        let (l1, l2) = get_sample_vectors::<f64>(1043);
        unsafe { crate::danger::op_min::test_min::<_, Avx2>(l1, l2) };
    }

    #[test]
    fn test_sum() {
        let (l1, l2) = (vec![1.0f32; 1043], vec![3.0f32; 1043]);
        unsafe { crate::danger::op_sum::test_sum::<_, Avx2>(l1, l2) };

        let (l1, l2) = (vec![1.0f32; 1043], vec![3.0f32; 1043]);
        unsafe { crate::danger::op_sum::test_sum::<_, Avx2>(l1, l2) };
    }

    #[test]
    fn test_vector_x_value() {
        let (l1, _) = (vec![1.0f32; 1043], vec![3.0f32; 1043]);
        unsafe {
            crate::danger::op_vector_x_value::tests::test_add::<_, Avx2>(l1.clone(), 2.0)
        };
        unsafe {
            crate::danger::op_vector_x_value::tests::test_sub::<_, Avx2>(l1.clone(), 2.0)
        };
        unsafe {
            crate::danger::op_vector_x_value::tests::test_div::<_, Avx2>(l1.clone(), 2.0)
        };
        unsafe { crate::danger::op_vector_x_value::tests::test_mul::<_, Avx2>(l1, 2.0) };
    }

    #[test]
    fn test_vector_x_vector() {
        let (l1, l2) = (vec![1.0f32; 1043], vec![3.0f32; 1043]);
        unsafe {
            crate::danger::op_vector_x_vector::tests::test_add::<_, Avx2>(
                l1.clone(),
                l2.clone(),
            )
        };
        unsafe {
            crate::danger::op_vector_x_vector::tests::test_sub::<_, Avx2>(
                l1.clone(),
                l2.clone(),
            )
        };
        unsafe {
            crate::danger::op_vector_x_vector::tests::test_div::<_, Avx2>(
                l1.clone(),
                l2.clone(),
            )
        };
        unsafe { crate::danger::op_vector_x_vector::tests::test_mul::<_, Avx2>(l1, l2) };
    }
}
