use crate::buffer::WriteOnlyBuffer;
use crate::danger::core_simd_api::{DenseLane, SimdRegister};
use crate::math::Math;

#[inline(always)]
/// A generic vector addition implementation over one vector and single value.
///
/// # Safety
///
/// The sizes of `a` and `result` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_add_value<T, R, M, B>(
    dims: usize,
    value: T,
    a: &[T],
    mut result: B,
) where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
    B: WriteOnlyBuffer<Item = T>,
{
    debug_assert_eq!(a.len(), dims, "Vector a does not match size `dims`");
    debug_assert_eq!(
        result.raw_buffer_len(),
        dims,
        "Vector result does not match size `dims`"
    );

    let value_reg = R::filled(value);
    let value_dense = DenseLane::copy(value_reg);

    let offset_from = dims % R::elements_per_dense();
    let a_ptr = a.as_ptr();
    let result_ptr = result.as_write_only_ptr();

    // Operate over dense lanes first.
    let mut i = 0;
    while i < (dims - offset_from) {
        let l1 = R::load_dense(a_ptr.add(i));
        let sum = R::add_dense(l1, value_dense);
        R::write_dense(result_ptr.add(i), sum);

        i += R::elements_per_dense();
    }

    // Operate over single registers next.
    let offset_from = offset_from % R::elements_per_lane();
    while i < (dims - offset_from) {
        let l1 = R::load(a_ptr.add(i));
        let sum = R::add(l1, value_reg);
        R::write(result_ptr.add(i), sum);

        i += R::elements_per_lane();
    }

    while i < dims {
        let a = *a.get_unchecked(i);
        result.write_at(i, M::add(a, value));

        i += 1;
    }
}

#[inline(always)]
/// A generic vector subtraction implementation over one vector and single value.
///
/// # Safety
///
/// The sizes of `a` and `result` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_sub_value<T, R, M, B>(
    dims: usize,
    value: T,
    a: &[T],
    mut result: B,
) where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
    B: WriteOnlyBuffer<Item = T>,
{
    debug_assert_eq!(a.len(), dims, "Vector a does not match size `dims`");
    debug_assert_eq!(
        result.raw_buffer_len(),
        dims,
        "Vector result does not match size `dims`"
    );

    let value_reg = R::filled(value);
    let value_dense = DenseLane::copy(value_reg);

    let offset_from = dims % R::elements_per_dense();
    let a_ptr = a.as_ptr();
    let result_ptr = result.as_write_only_ptr();

    // Operate over dense lanes first.
    let mut i = 0;
    while i < (dims - offset_from) {
        let l1 = R::load_dense(a_ptr.add(i));
        let sum = R::sub_dense(l1, value_dense);
        R::write_dense(result_ptr.add(i), sum);

        i += R::elements_per_dense();
    }

    // Operate over single registers next.
    let offset_from = offset_from % R::elements_per_lane();
    while i < (dims - offset_from) {
        let l1 = R::load(a_ptr.add(i));
        let sum = R::sub(l1, value_reg);
        R::write(result_ptr.add(i), sum);

        i += R::elements_per_lane();
    }

    while i < dims {
        let a = *a.get_unchecked(i);
        result.write_at(i, M::sub(a, value));

        i += 1;
    }
}

#[inline(always)]
/// A generic vector multiplication implementation over one vector and single value.
///
/// # Safety
///
/// The sizes of `a` and `result` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_mul_value<T, R, M, B>(
    dims: usize,
    value: T,
    a: &[T],
    mut result: B,
) where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
    B: WriteOnlyBuffer<Item = T>,
{
    debug_assert_eq!(a.len(), dims, "Vector a does not match size `dims`");
    debug_assert_eq!(
        result.raw_buffer_len(),
        dims,
        "Vector result does not match size `dims`"
    );

    let value_reg = R::filled(value);
    let value_dense = DenseLane::copy(value_reg);

    let offset_from = dims % R::elements_per_dense();
    let a_ptr = a.as_ptr();
    let result_ptr = result.as_write_only_ptr();

    // Operate over dense lanes first.
    let mut i = 0;
    while i < (dims - offset_from) {
        let l1 = R::load_dense(a_ptr.add(i));
        let sum = R::mul_dense(l1, value_dense);
        R::write_dense(result_ptr.add(i), sum);

        i += R::elements_per_dense();
    }

    // Operate over single registers next.
    let offset_from = offset_from % R::elements_per_lane();
    while i < (dims - offset_from) {
        let l1 = R::load(a_ptr.add(i));
        let sum = R::mul(l1, value_reg);
        R::write(result_ptr.add(i), sum);

        i += R::elements_per_lane();
    }

    while i < dims {
        let a = *a.get_unchecked(i);
        result.write_at(i, M::mul(a, value));

        i += 1;
    }
}

#[inline(always)]
/// A generic vector division implementation over one vector and single value.
///
/// NOTE:
/// This method _DOES NOT CHEAT_ meaning it will not convert the value to be `1 / value`
/// and use a multiply operation, instead you must do this yourself if you wish for
/// the 'smarter'/faster version.
///
/// # Safety
///
/// The sizes of `a` and `result` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_div_value<T, R, M, B>(
    dims: usize,
    value: T,
    a: &[T],
    mut result: B,
) where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
    B: WriteOnlyBuffer<Item = T>,
{
    debug_assert_eq!(a.len(), dims, "Vector a does not match size `dims`");
    debug_assert_eq!(
        result.raw_buffer_len(),
        dims,
        "Vector result does not match size `dims`"
    );

    let value_reg = R::filled(value);
    let value_dense = DenseLane::copy(value_reg);

    let offset_from = dims % R::elements_per_dense();
    let a_ptr = a.as_ptr();
    let result_ptr = result.as_write_only_ptr();

    // Operate over dense lanes first.
    let mut i = 0;
    while i < (dims - offset_from) {
        let l1 = R::load_dense(a_ptr.add(i));
        let sum = R::div_dense(l1, value_dense);
        R::write_dense(result_ptr.add(i), sum);

        i += R::elements_per_dense();
    }

    // Operate over single registers next.
    let offset_from = offset_from % R::elements_per_lane();
    while i < (dims - offset_from) {
        let l1 = R::load(a_ptr.add(i));
        let sum = R::div(l1, value_reg);
        R::write(result_ptr.add(i), sum);

        i += R::elements_per_lane();
    }

    while i < dims {
        let a = *a.get_unchecked(i);
        result.write_at(i, M::div(a, value));

        i += 1;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::danger::SimdRegister;
    use crate::math::Math;

    pub(crate) unsafe fn test_add<T, R>(l1: Vec<T>, value: T)
    where
        T: Copy + PartialEq + std::fmt::Debug,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut Vec<T>: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_add_value::<T, R, AutoMath, _>(dims, value, &l1, &mut result);

        let mut expected_result = Vec::new();
        for a in l1 {
            expected_result.push(AutoMath::add(a, value));
        }
        assert_eq!(result, expected_result, "value missmatch");
    }

    pub(crate) unsafe fn test_sub<T, R>(l1: Vec<T>, value: T)
    where
        T: Copy + PartialEq + std::fmt::Debug,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut Vec<T>: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_sub_value::<T, R, AutoMath, _>(dims, value, &l1, &mut result);

        let mut expected_result = Vec::new();
        for a in l1 {
            expected_result.push(AutoMath::sub(a, value));
        }
        assert_eq!(result, expected_result, "value missmatch");
    }

    pub(crate) unsafe fn test_div<T, R>(l1: Vec<T>, value: T)
    where
        T: Copy + PartialEq + std::fmt::Debug,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut Vec<T>: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_div_value::<T, R, AutoMath, _>(dims, value, &l1, &mut result);

        let mut expected_result = Vec::new();
        for a in l1 {
            expected_result.push(AutoMath::div(a, value));
        }
        assert_eq!(result, expected_result, "value missmatch");
    }

    pub(crate) unsafe fn test_mul<T, R>(l1: Vec<T>, value: T)
    where
        T: Copy + PartialEq + std::fmt::Debug,
        R: SimdRegister<T>,
        crate::math::AutoMath: Math<T>,
        for<'a> &'a mut Vec<T>: WriteOnlyBuffer<Item = T>,
    {
        use crate::math::AutoMath;

        let dims = l1.len();
        let mut result = vec![AutoMath::zero(); dims];
        generic_mul_value::<T, R, AutoMath, _>(dims, value, &l1, &mut result);

        let mut expected_result = Vec::new();
        for a in l1 {
            expected_result.push(AutoMath::mul(a, value));
        }
        assert_eq!(result, expected_result, "value missmatch");
    }
}
