use super::core_routine_boilerplate::apply_vertical_kernel;
use crate::buffer::WriteOnlyBuffer;
use crate::danger::SimdRegister;
use crate::math::Math;
use crate::mem_loader::{IntoMemLoader, MemLoader};

#[inline(always)]
/// A generic vector element-wise equality check of vectors `a` and `b` checking if
/// element of `a` is **_equal to_** element of `b`.
///
/// The result of each element check is returned as a mask of either `0` (false) or ` 1` (true).
///
/// # Safety
///
/// The sizes of `a`, `b` and `result` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_cmp_eq_vertical<T, R, M, B1, B2, B3>(
    a: B1,
    b: B2,
    result: &mut [B3],
) where
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
        R::eq_dense,
        R::eq,
        |a, b| M::cast_bool(M::cmp_eq(a, b)),
    )
}

#[inline(always)]
/// A generic vector element-wise equality check of vectors `a` and `b` checking if
/// element of `a` is **_not equal to_** element of `b`.
///
/// The result of each element check is returned as a mask of either `0` (false) or ` 1` (true).
///
/// # Safety
///
/// The sizes of `a`, `b` and `result` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_cmp_neq_vertical<T, R, M, B1, B2, B3>(
    a: B1,
    b: B2,
    result: &mut [B3],
) where
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
        R::neq_dense,
        R::neq,
        |a, b| M::cast_bool(!M::cmp_eq(a, b)),
    )
}

#[inline(always)]
/// A generic vector element-wise equality check of vectors `a` and `b` checking if
/// element of `a` is **_less than_** element of `b`.
///
/// The result of each element check is returned as a mask of either `0` (false) or ` 1` (true).
///
/// # Safety
///
/// The sizes of `a`, `b` and `result` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_cmp_lt_vertical<T, R, M, B1, B2, B3>(
    a: B1,
    b: B2,
    result: &mut [B3],
) where
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
        R::lt_dense,
        R::lt,
        |a, b| M::cast_bool(M::cmp_lt(a, b)),
    )
}

#[inline(always)]
/// A generic vector element-wise equality check of vectors `a` and `b` checking if
/// element of `a` is **_less than or equal to_** element of `b`.
///
/// The result of each element check is returned as a mask of either `0` (false) or ` 1` (true).
///
/// # Safety
///
/// The sizes of `a`, `b` and `result` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_cmp_lte_vertical<T, R, M, B1, B2, B3>(
    a: B1,
    b: B2,
    result: &mut [B3],
) where
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
        R::lte_dense,
        R::lte,
        |a, b| M::cast_bool(M::cmp_lte(a, b)),
    )
}

#[inline(always)]
/// A generic vector element-wise equality check of vectors `a` and `b` checking if
/// element of `a` is **_greater than_** element of `b`.
///
/// The result of each element check is returned as a mask of either `0` (false) or ` 1` (true).
///
/// # Safety
///
/// The sizes of `a`, `b` and `result` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_cmp_gt_vertical<T, R, M, B1, B2, B3>(
    a: B1,
    b: B2,
    result: &mut [B3],
) where
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
        R::gt_dense,
        R::gt,
        |a, b| M::cast_bool(M::cmp_gt(a, b)),
    )
}

#[inline(always)]
/// A generic vector element-wise equality check of vectors `a` and `b` checking if
/// element of `a` is **_greater than or equal to_** element of `b`.
///
/// The result of each element check is returned as a mask of either `0` (false) or ` 1` (true).
///
/// # Safety
///
/// The sizes of `a`, `b` and `result` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_cmp_gte_vertical<T, R, M, B1, B2, B3>(
    a: B1,
    b: B2,
    result: &mut [B3],
) where
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
        R::gte_dense,
        R::gte,
        |a, b| M::cast_bool(M::cmp_gte(a, b)),
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use std::iter::zip;

    use super::*;
    use crate::danger::SimdRegister;
    use crate::math::Math;

    pub(crate) unsafe fn test_simple_vectors_eq<T, R>(l1: Vec<T>, l2: Vec<T>)
    where
        T: Copy + PartialEq + std::fmt::Debug,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_cmp_eq_vertical::<T, R, AutoMath, _, _, _>(&l1, &l2, &mut result);

        let mut expected_result = Vec::new();
        for (a, b) in zip(l1, l2) {
            expected_result.push(AutoMath::cast_bool(AutoMath::cmp_eq(a, b)));
        }
        assert_eq!(result, expected_result, "value mismatch");
    }

    pub(crate) unsafe fn test_simple_vectors_neq<T, R>(l1: Vec<T>, l2: Vec<T>)
    where
        T: Copy + PartialEq + std::fmt::Debug,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_cmp_neq_vertical::<T, R, AutoMath, _, _, _>(&l1, &l2, &mut result);

        let mut expected_result = Vec::new();
        for (a, b) in zip(l1, l2) {
            expected_result.push(AutoMath::cast_bool(!AutoMath::cmp_eq(a, b)));
        }
        assert_eq!(result, expected_result, "value mismatch");
    }

    pub(crate) unsafe fn test_simple_vectors_lt<T, R>(l1: Vec<T>, l2: Vec<T>)
    where
        T: Copy + PartialEq + std::fmt::Debug,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_cmp_lt_vertical::<T, R, AutoMath, _, _, _>(&l1, &l2, &mut result);

        let mut expected_result = Vec::new();
        for (a, b) in zip(l1, l2) {
            expected_result.push(AutoMath::cast_bool(AutoMath::cmp_lt(a, b)));
        }
        assert_eq!(result, expected_result, "value mismatch");
    }

    pub(crate) unsafe fn test_simple_vectors_lte<T, R>(l1: Vec<T>, l2: Vec<T>)
    where
        T: Copy + PartialEq + std::fmt::Debug,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_cmp_lte_vertical::<T, R, AutoMath, _, _, _>(&l1, &l2, &mut result);

        let mut expected_result = Vec::new();
        for (a, b) in zip(l1, l2) {
            expected_result.push(AutoMath::cast_bool(AutoMath::cmp_lte(a, b)));
        }
        assert_eq!(result, expected_result, "value mismatch");
    }

    pub(crate) unsafe fn test_simple_vectors_gt<T, R>(l1: Vec<T>, l2: Vec<T>)
    where
        T: Copy + PartialEq + std::fmt::Debug,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_cmp_gt_vertical::<T, R, AutoMath, _, _, _>(&l1, &l2, &mut result);

        let mut expected_result = Vec::new();
        for (a, b) in zip(l1, l2) {
            expected_result.push(AutoMath::cast_bool(AutoMath::cmp_gt(a, b)));
        }
        assert_eq!(result, expected_result, "value mismatch");
    }

    pub(crate) unsafe fn test_simple_vectors_gte<T, R>(l1: Vec<T>, l2: Vec<T>)
    where
        T: Copy + PartialEq + std::fmt::Debug,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_cmp_gte_vertical::<T, R, AutoMath, _, _, _>(&l1, &l2, &mut result);

        let mut expected_result = Vec::new();
        for (a, b) in zip(l1, l2) {
            expected_result.push(AutoMath::cast_bool(AutoMath::cmp_gte(a, b)));
        }
        assert_eq!(result, expected_result, "value mismatch");
    }
    
    // Broadcast value tests
    pub(crate) unsafe fn test_broadcast_value_eq<T, R>(l1: Vec<T>, value: T)
    where
        T: Copy + PartialEq + std::fmt::Debug + IntoMemLoader<T>,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_cmp_eq_vertical::<T, R, AutoMath, _, _, _>(&l1, value, &mut result);

        let mut expected_result = Vec::new();
        for a in l1 {
            expected_result.push(AutoMath::cast_bool(AutoMath::cmp_eq(a, value)));
        }
        assert_eq!(result, expected_result, "value mismatch");
    }

    pub(crate) unsafe fn test_broadcast_value_neq<T, R>(l1: Vec<T>, value: T)
    where
        T: Copy + PartialEq + std::fmt::Debug + IntoMemLoader<T>,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_cmp_neq_vertical::<T, R, AutoMath, _, _, _>(&l1, value, &mut result);

        let mut expected_result = Vec::new();
        for a in l1 {
            expected_result.push(AutoMath::cast_bool(!AutoMath::cmp_eq(a, value)));
        }
        assert_eq!(result, expected_result, "value mismatch");
    }

    pub(crate) unsafe fn test_broadcast_value_lt<T, R>(l1: Vec<T>, value: T)
    where
        T: Copy + PartialEq + std::fmt::Debug + IntoMemLoader<T>,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_cmp_lt_vertical::<T, R, AutoMath, _, _, _>(&l1, value, &mut result);

        let mut expected_result = Vec::new();
        for a in l1 {
            expected_result.push(AutoMath::cast_bool(AutoMath::cmp_lt(a, value)));
        }
        assert_eq!(result, expected_result, "value mismatch");
    }

    pub(crate) unsafe fn test_broadcast_value_lte<T, R>(l1: Vec<T>, value: T)
    where
        T: Copy + PartialEq + std::fmt::Debug + IntoMemLoader<T>,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_cmp_lte_vertical::<T, R, AutoMath, _, _, _>(&l1, value, &mut result);

        let mut expected_result = Vec::new();
        for a in l1 {
            expected_result.push(AutoMath::cast_bool(AutoMath::cmp_lte(a, value)));
        }
        assert_eq!(result, expected_result, "value mismatch");
    }

    pub(crate) unsafe fn test_broadcast_value_gt<T, R>(l1: Vec<T>, value: T)
    where
        T: Copy + PartialEq + std::fmt::Debug + IntoMemLoader<T>,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_cmp_gt_vertical::<T, R, AutoMath, _, _, _>(&l1, value, &mut result);

        let mut expected_result = Vec::new();
        for a in l1 {
            expected_result.push(AutoMath::cast_bool(AutoMath::cmp_gt(a, value)));
        }
        assert_eq!(result, expected_result, "value mismatch");
    }

    pub(crate) unsafe fn test_broadcast_value_gte<T, R>(l1: Vec<T>, value: T)
    where
        T: Copy + PartialEq + std::fmt::Debug + IntoMemLoader<T>,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_cmp_gte_vertical::<T, R, AutoMath, _, _, _>(&l1, value, &mut result);

        let mut expected_result = Vec::new();
        for a in l1 {
            expected_result.push(AutoMath::cast_bool(AutoMath::cmp_gte(a, value)));
        }
        assert_eq!(result, expected_result, "value mismatch");
    }
}
