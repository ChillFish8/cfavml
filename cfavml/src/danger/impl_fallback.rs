use crate::danger::SimdRegister;
use crate::math::{AutoMath, Math};

/// Fallback SIMD-like operations.
///
/// This is designed to provide abstract operations that are easily optimized by the compiler
/// even if we're not manually writing the SIMD, hopefully to cover other architectures that
/// we haven't manually supported.
pub struct Fallback;

impl SimdRegister<f32> for Fallback {
    type Register = f32;

    #[inline(always)]
    unsafe fn load(mem: *const f32) -> Self::Register {
        mem.read()
    }

    #[inline(always)]
    unsafe fn filled(value: f32) -> Self::Register {
        value
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        0.0
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        AutoMath::add(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        AutoMath::sub(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        AutoMath::mul(l1, l2)
    }

    #[inline(always)]
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        AutoMath::div(l1, l2)
    }

    #[inline(always)]
    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        let res = AutoMath::mul(l1, l2);
        AutoMath::add(res, acc)
    }

    #[inline(always)]
    unsafe fn max(l1: Self::Register, l2: Self::Register) -> Self::Register {
        AutoMath::cmp_max(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        AutoMath::cmp_min(l1, l2)
    }

    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> f32 {
        reg
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> f32 {
        reg
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> f32 {
        reg
    }

    #[inline(always)]
    unsafe fn write(mem: *mut f32, reg: Self::Register) {
        mem.write(reg)
    }
}

impl SimdRegister<f64> for Fallback {
    type Register = f64;

    #[inline(always)]
    unsafe fn load(mem: *const f64) -> Self::Register {
        mem.read()
    }

    #[inline(always)]
    unsafe fn filled(value: f64) -> Self::Register {
        value
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        0.0
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        AutoMath::add(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        AutoMath::sub(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        AutoMath::mul(l1, l2)
    }

    #[inline(always)]
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        AutoMath::div(l1, l2)
    }

    #[inline(always)]
    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        let res = AutoMath::mul(l1, l2);
        AutoMath::add(res, acc)
    }

    #[inline(always)]
    unsafe fn max(l1: Self::Register, l2: Self::Register) -> Self::Register {
        AutoMath::cmp_max(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        AutoMath::cmp_min(l1, l2)
    }

    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> f64 {
        reg
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> f64 {
        reg
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> f64 {
        reg
    }

    #[inline(always)]
    unsafe fn write(mem: *mut f64, reg: Self::Register) {
        mem.write(reg)
    }
}

impl SimdRegister<i32> for Fallback {
    type Register = i32;

    #[inline(always)]
    unsafe fn load(mem: *const i32) -> Self::Register {
        mem.read()
    }

    #[inline(always)]
    unsafe fn filled(value: i32) -> Self::Register {
        value
    }

    #[inline(always)]
    unsafe fn zeroed() -> Self::Register {
        0
    }

    #[inline(always)]
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register {
        AutoMath::add(l1, l2)
    }

    #[inline(always)]
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register {
        AutoMath::sub(l1, l2)
    }

    #[inline(always)]
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register {
        AutoMath::mul(l1, l2)
    }

    #[inline(always)]
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register {
        AutoMath::div(l1, l2)
    }

    #[inline(always)]
    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register {
        let res = AutoMath::mul(l1, l2);
        AutoMath::add(res, acc)
    }

    #[inline(always)]
    unsafe fn max(l1: Self::Register, l2: Self::Register) -> Self::Register {
        AutoMath::cmp_max(l1, l2)
    }

    #[inline(always)]
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register {
        AutoMath::cmp_min(l1, l2)
    }

    #[inline(always)]
    unsafe fn sum_to_value(reg: Self::Register) -> i32 {
        reg
    }

    #[inline(always)]
    unsafe fn max_to_value(reg: Self::Register) -> i32 {
        reg
    }

    #[inline(always)]
    unsafe fn min_to_value(reg: Self::Register) -> i32 {
        reg
    }

    #[inline(always)]
    unsafe fn write(mem: *mut i32, reg: Self::Register) {
        mem.write(reg)
    }
}


#[cfg(test)]
mod tests {
    use std::fmt::Debug;
    use super::*;
    use crate::test_utils::get_sample_vectors;

    #[test]
    fn test_suite() {
        unsafe { crate::danger::impl_test::test_suite_impl_f32::<Fallback>() }
    }

    #[test]
    fn test_cosine() {
        let (l1, l2) = get_sample_vectors::<f32>(1043);
        unsafe { crate::danger::op_cosine::test_cosine::<_, Fallback>(l1, l2) };

        let (l1, l2) = get_sample_vectors::<f64>(1043);
        unsafe { crate::danger::op_cosine::test_cosine::<_, Fallback>(l1, l2) };

        let (l1, l2) = get_sample_vectors::<i32>(1043);
        unsafe { crate::danger::op_cosine::test_cosine::<_, Fallback>(l1, l2) };
    }

    #[test]
    fn test_dot_product() {
        let (l1, l2) = get_sample_vectors::<f32>(1043);
        unsafe { crate::danger::op_dot_product::test_dot::<_, Fallback>(l1, l2) };

        let (l1, l2) = get_sample_vectors::<f64>(1043);
        unsafe { crate::danger::op_dot_product::test_dot::<_, Fallback>(l1, l2) };

        let (l1, l2) = get_sample_vectors::<i32>(1043);
        unsafe { crate::danger::op_dot_product::test_dot::<_, Fallback>(l1, l2) };
    }

    #[test]
    fn test_norm() {
        let (l1, _) = get_sample_vectors::<f32>(1043);
        unsafe { crate::danger::op_norm::test_norm::<_, Fallback>(l1) };

        let (l1, _) = get_sample_vectors::<f64>(1043);
        unsafe { crate::danger::op_norm::test_norm::<_, Fallback>(l1) };

        let (l1, _) = get_sample_vectors::<i32>(1043);
        unsafe { crate::danger::op_norm::test_norm::<_, Fallback>(l1) };
    }

    #[test]
    fn test_euclidean() {
        let (l1, l2) = get_sample_vectors::<f32>(1043);
        unsafe { crate::danger::op_euclidean::test_euclidean::<_, Fallback>(l1, l2) };

        let (l1, l2) = get_sample_vectors::<f64>(1043);
        unsafe { crate::danger::op_euclidean::test_euclidean::<_, Fallback>(l1, l2) };

        let (l1, l2) = get_sample_vectors::<i32>(1043);
        unsafe { crate::danger::op_euclidean::test_euclidean::<_, Fallback>(l1, l2) };
    }

    #[test]
    fn test_max() {
        let (l1, l2) = get_sample_vectors::<f32>(1043);
        unsafe { crate::danger::op_max::test_max::<_, Fallback>(l1, l2) };

        let (l1, l2) = get_sample_vectors::<f64>(1043);
        unsafe { crate::danger::op_max::test_max::<_, Fallback>(l1, l2) };

        let (l1, l2) = get_sample_vectors::<i32>(1043);
        unsafe { crate::danger::op_max::test_max::<_, Fallback>(l1, l2) };
    }

    #[test]
    fn test_min() {
        let (l1, l2) = get_sample_vectors::<f32>(1043);
        unsafe { crate::danger::op_min::test_min::<_, Fallback>(l1, l2) };

        let (l1, l2) = get_sample_vectors::<f64>(1043);
        unsafe { crate::danger::op_min::test_min::<_, Fallback>(l1, l2) };

        let (l1, l2) = get_sample_vectors::<i32>(1043);
        unsafe { crate::danger::op_min::test_min::<_, Fallback>(l1, l2) };
    }

    #[test]
    fn test_sum() {
        let (l1, l2) = (vec![1.0f32; 1043], vec![3.0f32; 1043]);
        unsafe { crate::danger::op_sum::test_sum::<_, Fallback>(l1, l2) };

        let (l1, l2) = (vec![1.0f64; 1043], vec![3.0f64; 1043]);
        unsafe { crate::danger::op_sum::test_sum::<_, Fallback>(l1, l2) };

        let (l1, l2) = (vec![1i32; 1043], vec![3i32; 1043]);
        unsafe { crate::danger::op_sum::test_sum::<_, Fallback>(l1, l2) };
    }

    #[test]
    fn test_vector_x_value() {
        let (l1, _) = (vec![1.0f32; 1043], vec![3.0f32; 1043]);
        test_vector_x_value_all(l1, 2.0);

        let (l1, _) = (vec![1.0f64; 1043], vec![3.0f64; 1043]);
        test_vector_x_value_all(l1, 2.0);

        let (l1, _) = (vec![1i32; 1043], vec![3i32; 1043]);
        test_vector_x_value_all(l1, 2);
    }

    #[test]
    fn test_vector_x_vector() {
        let (l1, l2) = (vec![1.0f32; 1043], vec![3.0f32; 1043]);
        test_vector_x_vector_all(l1, l2);

        let (l1, l2) = (vec![1.0f64; 1043], vec![3.0f64; 1043]);
        test_vector_x_vector_all(l1, l2);

        let (l1, l2) = (vec![1i32; 1043], vec![3i32; 1043]);
        test_vector_x_vector_all(l1, l2);
    }

    fn test_vector_x_value_all<T: Copy + PartialEq + Debug>(l1: Vec<T>, value: T)
        where
            Fallback: SimdRegister<T>,
            AutoMath: Math<T>,
    {
        unsafe {
            crate::danger::op_vector_x_value::tests::test_add::<_, Fallback>(
                l1.clone(),
                value,
            );
            crate::danger::op_vector_x_value::tests::test_sub::<_, Fallback>(
                l1.clone(),
                value,
            );
            crate::danger::op_vector_x_value::tests::test_div::<_, Fallback>(
                l1.clone(),
                value,
            );
            crate::danger::op_vector_x_value::tests::test_mul::<_, Fallback>(l1, value)
        };
    }

    fn test_vector_x_vector_all<T: Copy + PartialEq + Debug>(l1: Vec<T>, l2: Vec<T>)
    where
        Fallback: SimdRegister<T>,
        AutoMath: Math<T>,
    {
        unsafe {
            crate::danger::op_vector_x_vector::tests::test_add::<_, Fallback>(
                l1.clone(),
                l2.clone(),
            );
            crate::danger::op_vector_x_vector::tests::test_sub::<_, Fallback>(
                l1.clone(),
                l2.clone(),
            );
            crate::danger::op_vector_x_vector::tests::test_div::<_, Fallback>(
                l1.clone(),
                l2.clone(),
            );
            crate::danger::op_vector_x_vector::tests::test_mul::<_, Fallback>(l1, l2);
        };
    }
}
