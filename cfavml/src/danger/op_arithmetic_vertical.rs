use super::core_routine_boilerplate::apply_vertical_kernel;
use super::core_simd_api::SimdRegister;
use crate::buffer::WriteOnlyBuffer;
use crate::math::Math;
use crate::mem_loader::{IntoMemLoader, MemLoader};

#[inline(always)]
/// A generic vector addition implementation over one vector and single value.
///
/// # Safety
///
/// The sizes of `a`, `b` and `result` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_add_vertical<T, R, M, B1, B2, B3>(a: B1, b: B2, result: &mut [B3])
where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
    B1: IntoMemLoader<T>,
    B1::Loader: MemLoader<Value = T>,
    B2: IntoMemLoader<T>,
    B2::Loader: MemLoader<Value = T>,
    for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = T>,
{
    apply_vertical_kernel::<T, R, M, B1, B2, B3>(
        a,
        b,
        result,
        R::add_dense,
        R::add,
        M::add,
    )
}

#[inline(always)]
/// A generic vector subtraction implementation over one vector and single value.
///
/// # Safety
///
/// The sizes of `a`, `b` and `result` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_sub_vertical<T, R, M, B1, B2, B3>(a: B1, b: B2, result: &mut [B3])
where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
    B1: IntoMemLoader<T>,
    B1::Loader: MemLoader<Value = T>,
    B2: IntoMemLoader<T>,
    B2::Loader: MemLoader<Value = T>,
    for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = T>,
{
    apply_vertical_kernel::<T, R, M, B1, B2, B3>(
        a,
        b,
        result,
        R::sub_dense,
        R::sub,
        M::sub,
    )
}

#[inline(always)]
/// A generic vector multiplication implementation over one vector and single value.
///
/// # Safety
///
/// The sizes of `a`, `b` and `result` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_mul_vertical<T, R, M, B1, B2, B3>(a: B1, b: B2, result: &mut [B3])
where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
    B1: IntoMemLoader<T>,
    B1::Loader: MemLoader<Value = T>,
    B2: IntoMemLoader<T>,
    B2::Loader: MemLoader<Value = T>,
    for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = T>,
{
    apply_vertical_kernel::<T, R, M, B1, B2, B3>(
        a,
        b,
        result,
        R::mul_dense,
        R::mul,
        M::mul,
    )
}

#[inline(always)]
/// A generic vector division implementation dividing by vector `b`.
///
/// # Safety
///
/// The sizes of `a`, `b` and `result` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_div_vertical<T, R, M, B1, B2, B3>(a: B1, b: B2, result: &mut [B3])
where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
    B1: IntoMemLoader<T>,
    B1::Loader: MemLoader<Value = T>,
    B2: IntoMemLoader<T>,
    B2::Loader: MemLoader<Value = T>,
    for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = T>,
{
    apply_vertical_kernel::<T, R, M, B1, B2, B3>(
        a,
        b,
        result,
        R::div_dense,
        R::div,
        M::div,
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::danger::SimdRegister;
    use crate::math::Math;

    pub(crate) unsafe fn test_add<T, R>(l1: Vec<T>, l2: Vec<T>)
    where
        T: Copy + PartialEq + std::fmt::Debug,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_add_vertical::<T, R, AutoMath, _, _, _>(&l1, &l2, &mut result);

        let mut expected_result = Vec::new();
        for (a, b) in l1.iter().copied().zip(l2) {
            expected_result.push(AutoMath::add(a, b));
        }
        assert_eq!(result, expected_result, "value mismatch");
    }

    pub(crate) unsafe fn test_sub<T, R>(l1: Vec<T>, l2: Vec<T>)
    where
        T: Copy + PartialEq + std::fmt::Debug,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_sub_vertical::<T, R, AutoMath, _, _, _>(&l1, &l2, &mut result);

        let mut expected_result = Vec::new();
        for (a, b) in l1.iter().copied().zip(l2) {
            expected_result.push(AutoMath::sub(a, b));
        }
        assert_eq!(result, expected_result, "value mismatch");
    }

    pub(crate) unsafe fn test_div<T, R>(l1: Vec<T>, l2: Vec<T>)
    where
        T: Copy + PartialEq + std::fmt::Debug,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_div_vertical::<T, R, AutoMath, _, _, _>(&l1, &l2, &mut result);

        let mut expected_result = Vec::new();
        for (a, b) in l1.iter().copied().zip(l2) {
            expected_result.push(AutoMath::div(a, b));
        }
        assert_eq!(result, expected_result, "value mismatch");
    }

    pub(crate) unsafe fn test_mul<T, R>(l1: Vec<T>, l2: Vec<T>)
    where
        T: Copy + PartialEq + std::fmt::Debug,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_mul_vertical::<T, R, AutoMath, _, _, _>(&l1, &l2, &mut result);

        let mut expected_result = Vec::new();
        for (a, b) in l1.iter().copied().zip(l2) {
            expected_result.push(AutoMath::mul(a, b));
        }
        assert_eq!(result, expected_result, "value mismatch");
    }
}
