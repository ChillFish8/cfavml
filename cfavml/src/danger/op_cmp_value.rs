use super::core_routine_boilerplate::apply_vector_x_value_kernel;
use crate::buffer::WriteOnlyBuffer;
use crate::danger::SimdRegister;
use crate::math::Math;

#[inline(always)]
/// A generic vector equality check against a broadcast value to see if each element of `a` is
/// **_equal to_** `value`.
///
/// The result of each element check is returned as a mask of either `0` (false) or ` 1` (true).
///
/// # Safety
///
/// The sizes of `a` and `result` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_cmp_eq_value<T, R, M, B>(
    dims: usize,
    value: T,
    a: &[T],
    result: &mut [B],
) where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
    for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
{
    apply_vector_x_value_kernel::<T, R, B>(
        dims,
        value,
        a,
        result,
        R::eq_dense,
        R::eq,
        |a, b| M::cast_bool(M::cmp_eq(a, b)),
    )
}

#[inline(always)]
/// A generic vector equality check against a broadcast value to see if each element of `a` is
/// **_not equal to_** `value`.
///
/// The result of each element check is returned as a mask of either `0` (false) or ` 1` (true).
///
/// # Safety
///
/// The sizes of `a` and `result` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_cmp_neq_value<T, R, M, B>(
    dims: usize,
    value: T,
    a: &[T],
    result: &mut [B],
) where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
    for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
{
    apply_vector_x_value_kernel::<T, R, B>(
        dims,
        value,
        a,
        result,
        R::neq_dense,
        R::neq,
        |a, b| M::cast_bool(!M::cmp_eq(a, b)),
    )
}

#[inline(always)]
/// A generic vector equality check against a broadcast value to see if each element of `a` is
/// **_less than_** `value`.
///
/// The result of each element check is returned as a mask of either `0` (false) or ` 1` (true).
///
/// # Safety
///
/// The sizes of `a` and `result` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_cmp_lt_value<T, R, M, B>(
    dims: usize,
    value: T,
    a: &[T],
    result: &mut [B],
) where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
    for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
{
    apply_vector_x_value_kernel::<T, R, B>(
        dims,
        value,
        a,
        result,
        R::lt_dense,
        R::lt,
        |a, b| M::cast_bool(M::cmp_lt(a, b)),
    )
}

#[inline(always)]
/// A generic vector equality check against a broadcast value to see if each element of `a` is
/// **_less than or equal to_** `value`.
///
/// The result of each element check is returned as a mask of either `0` (false) or ` 1` (true).
///
/// # Safety
///
/// The sizes of `a` and `result` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_cmp_lte_value<T, R, M, B>(
    dims: usize,
    value: T,
    a: &[T],
    result: &mut [B],
) where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
    for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
{
    apply_vector_x_value_kernel::<T, R, B>(
        dims,
        value,
        a,
        result,
        R::lte_dense,
        R::lte,
        |a, b| M::cast_bool(M::cmp_lte(a, b)),
    )
}

#[inline(always)]
/// A generic vector equality check against a broadcast value to see if each element of `a` is
/// **_greater than_** `value`.
///
/// The result of each element check is returned as a mask of either `0` (false) or ` 1` (true).
///
/// # Safety
///
/// The sizes of `a` and `result` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_cmp_gt_value<T, R, M, B>(
    dims: usize,
    value: T,
    a: &[T],
    result: &mut [B],
) where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
    for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
{
    apply_vector_x_value_kernel::<T, R, B>(
        dims,
        value,
        a,
        result,
        R::gt_dense,
        R::gt,
        |a, b| M::cast_bool(M::cmp_gt(a, b)),
    )
}

#[inline(always)]
/// A generic vector equality check against a broadcast value to see if each element of `a` is
/// **_greater than or equal to_** `value`.
///
/// The result of each element check is returned as a mask of either `0` (false) or ` 1` (true).
///
/// # Safety
///
/// The sizes of `a` and `result` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_cmp_gte_value<T, R, M, B>(
    dims: usize,
    value: T,
    a: &[T],
    result: &mut [B],
) where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
    for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
{
    apply_vector_x_value_kernel::<T, R, B>(
        dims,
        value,
        a,
        result,
        R::gte_dense,
        R::gte,
        |a, b| M::cast_bool(M::cmp_gte(a, b)),
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::danger::SimdRegister;
    use crate::math::Math;

    pub(crate) unsafe fn test_eq<T, R>(l1: Vec<T>, value: T)
    where
        T: Copy + PartialEq + std::fmt::Debug,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_cmp_eq_value::<T, R, AutoMath, _>(dims, value, &l1, &mut result);

        let mut expected_result = Vec::new();
        for a in l1 {
            expected_result.push(AutoMath::cast_bool(AutoMath::cmp_eq(a, value)));
        }
        assert_eq!(result, expected_result, "value mismatch");
    }

    pub(crate) unsafe fn test_neq<T, R>(l1: Vec<T>, value: T)
    where
        T: Copy + PartialEq + std::fmt::Debug,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_cmp_neq_value::<T, R, AutoMath, _>(dims, value, &l1, &mut result);

        let mut expected_result = Vec::new();
        for a in l1 {
            expected_result.push(AutoMath::cast_bool(!AutoMath::cmp_eq(a, value)));
        }
        assert_eq!(result, expected_result, "value mismatch");
    }

    pub(crate) unsafe fn test_lt<T, R>(l1: Vec<T>, value: T)
    where
        T: Copy + PartialEq + std::fmt::Debug,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_cmp_lt_value::<T, R, AutoMath, _>(dims, value, &l1, &mut result);

        let mut expected_result = Vec::new();
        for a in l1 {
            expected_result.push(AutoMath::cast_bool(AutoMath::cmp_lt(a, value)));
        }
        assert_eq!(result, expected_result, "value mismatch");
    }

    pub(crate) unsafe fn test_lte<T, R>(l1: Vec<T>, value: T)
    where
        T: Copy + PartialEq + std::fmt::Debug,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_cmp_lte_value::<T, R, AutoMath, _>(dims, value, &l1, &mut result);

        let mut expected_result = Vec::new();
        for a in l1 {
            expected_result.push(AutoMath::cast_bool(AutoMath::cmp_lte(a, value)));
        }
        assert_eq!(result, expected_result, "value mismatch");
    }

    pub(crate) unsafe fn test_gt<T, R>(l1: Vec<T>, value: T)
    where
        T: Copy + PartialEq + std::fmt::Debug,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_cmp_gt_value::<T, R, AutoMath, _>(dims, value, &l1, &mut result);

        let mut expected_result = Vec::new();
        for a in l1 {
            expected_result.push(AutoMath::cast_bool(AutoMath::cmp_gt(a, value)));
        }
        assert_eq!(result, expected_result, "value mismatch");
    }

    pub(crate) unsafe fn test_gte<T, R>(l1: Vec<T>, value: T)
    where
        T: Copy + PartialEq + std::fmt::Debug,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_cmp_gte_value::<T, R, AutoMath, _>(dims, value, &l1, &mut result);

        let mut expected_result = Vec::new();
        for a in l1 {
            expected_result.push(AutoMath::cast_bool(AutoMath::cmp_gte(a, value)));
        }
        assert_eq!(result, expected_result, "value mismatch");
    }
}
