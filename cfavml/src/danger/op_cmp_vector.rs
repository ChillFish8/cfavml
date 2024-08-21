use super::core_routine_boilerplate::apply_vector_x_vector_kernel;
use crate::buffer::WriteOnlyBuffer;
use crate::danger::SimdRegister;
use crate::math::Math;

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
pub unsafe fn generic_cmp_eq_vector<T, R, M, B>(
    dims: usize,
    a: &[T],
    b: &[T],
    result: &mut [B],
) where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
    for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
{
    apply_vector_x_vector_kernel::<T, R, B>(
        dims,
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
pub unsafe fn generic_cmp_neq_vector<T, R, M, B>(
    dims: usize,
    a: &[T],
    b: &[T],
    result: &mut [B],
) where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
    for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
{
    apply_vector_x_vector_kernel::<T, R, B>(
        dims,
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
pub unsafe fn generic_cmp_lt_vector<T, R, M, B>(
    dims: usize,
    a: &[T],
    b: &[T],
    result: &mut [B],
) where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
    for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
{
    apply_vector_x_vector_kernel::<T, R, B>(
        dims,
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
pub unsafe fn generic_cmp_lte_vector<T, R, M, B>(
    dims: usize,
    a: &[T],
    b: &[T],
    result: &mut [B],
) where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
    for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
{
    apply_vector_x_vector_kernel::<T, R, B>(
        dims,
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
pub unsafe fn generic_cmp_gt_vector<T, R, M, B>(
    dims: usize,
    a: &[T],
    b: &[T],
    result: &mut [B],
) where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
    for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
{
    apply_vector_x_vector_kernel::<T, R, B>(
        dims,
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
pub unsafe fn generic_cmp_gte_vector<T, R, M, B>(
    dims: usize,
    a: &[T],
    b: &[T],
    result: &mut [B],
) where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
    for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
{
    apply_vector_x_vector_kernel::<T, R, B>(
        dims,
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

    pub(crate) unsafe fn test_eq<T, R>(l1: Vec<T>, l2: Vec<T>)
    where
        T: Copy + PartialEq + std::fmt::Debug,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_cmp_eq_vector::<T, R, AutoMath, _>(dims, &l1, &l2, &mut result);

        let mut expected_result = Vec::new();
        for (a, b) in zip(l1, l2) {
            expected_result.push(AutoMath::cast_bool(AutoMath::cmp_eq(a, b)));
        }
        assert_eq!(result, expected_result, "value mismatch");
    }

    pub(crate) unsafe fn test_neq<T, R>(l1: Vec<T>, l2: Vec<T>)
    where
        T: Copy + PartialEq + std::fmt::Debug,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_cmp_neq_vector::<T, R, AutoMath, _>(dims, &l1, &l2, &mut result);

        let mut expected_result = Vec::new();
        for (a, b) in zip(l1, l2) {
            expected_result.push(AutoMath::cast_bool(!AutoMath::cmp_eq(a, b)));
        }
        assert_eq!(result, expected_result, "value mismatch");
    }

    pub(crate) unsafe fn test_lt<T, R>(l1: Vec<T>, l2: Vec<T>)
    where
        T: Copy + PartialEq + std::fmt::Debug,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_cmp_lt_vector::<T, R, AutoMath, _>(dims, &l1, &l2, &mut result);

        let mut expected_result = Vec::new();
        for (a, b) in zip(l1, l2) {
            expected_result.push(AutoMath::cast_bool(AutoMath::cmp_lt(a, b)));
        }
        assert_eq!(result, expected_result, "value mismatch");
    }

    pub(crate) unsafe fn test_lte<T, R>(l1: Vec<T>, l2: Vec<T>)
    where
        T: Copy + PartialEq + std::fmt::Debug,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_cmp_lte_vector::<T, R, AutoMath, _>(dims, &l1, &l2, &mut result);

        let mut expected_result = Vec::new();
        for (a, b) in zip(l1, l2) {
            expected_result.push(AutoMath::cast_bool(AutoMath::cmp_lte(a, b)));
        }
        assert_eq!(result, expected_result, "value mismatch");
    }

    pub(crate) unsafe fn test_gt<T, R>(l1: Vec<T>, l2: Vec<T>)
    where
        T: Copy + PartialEq + std::fmt::Debug,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_cmp_gt_vector::<T, R, AutoMath, _>(dims, &l1, &l2, &mut result);

        let mut expected_result = Vec::new();
        for (a, b) in zip(l1, l2) {
            expected_result.push(AutoMath::cast_bool(AutoMath::cmp_gt(a, b)));
        }
        assert_eq!(result, expected_result, "value mismatch");
    }

    pub(crate) unsafe fn test_gte<T, R>(l1: Vec<T>, l2: Vec<T>)
    where
        T: Copy + PartialEq + std::fmt::Debug,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_cmp_gte_vector::<T, R, AutoMath, _>(dims, &l1, &l2, &mut result);

        let mut expected_result = Vec::new();
        for (a, b) in zip(l1, l2) {
            expected_result.push(AutoMath::cast_bool(AutoMath::cmp_gte(a, b)));
        }
        assert_eq!(result, expected_result, "value mismatch");
    }
}
