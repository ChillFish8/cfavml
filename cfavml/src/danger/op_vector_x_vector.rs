use crate::buffer::WriteOnlyBuffer;
use crate::danger::core_simd_api::SimdRegister;
use crate::math::Math;

#[inline(always)]
/// A generic vector addition implementation over one vector and single value.
///
/// # Safety
///
/// The sizes of `a`, `b` and `result` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_add_vector<T, R, M, B>(
    dims: usize,
    a: &[T],
    b: &[T],
    mut result: &mut [B],
) where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
    for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
{
    debug_assert_eq!(a.len(), dims, "Vector a does not match size `dims`");
    debug_assert_eq!(b.len(), dims, "Vector a does not match size `dims`");
    debug_assert_eq!(
        result.raw_buffer_len(),
        dims,
        "Vector result does not match size `dims`"
    );

    let offset_from = dims % R::elements_per_dense();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let result_ptr = result.as_write_only_ptr();

    // Operate over dense lanes first.
    let mut i = 0;
    while i < (dims - offset_from) {
        let l1 = R::load_dense(a_ptr.add(i));
        let l2 = R::load_dense(b_ptr.add(i));
        let res = R::add_dense(l1, l2);
        R::write_dense(result_ptr.add(i), res);

        i += R::elements_per_dense();
    }

    // Operate over single registers next.
    let offset_from = offset_from % R::elements_per_lane();
    while i < (dims - offset_from) {
        let l1 = R::load(a_ptr.add(i));
        let l2 = R::load(b_ptr.add(i));
        let res = R::add(l1, l2);
        R::write(result_ptr.add(i), res);

        i += R::elements_per_lane();
    }

    while i < dims {
        let a = *a.get_unchecked(i);
        let b = *b.get_unchecked(i);
        result.write_at(i, M::add(a, b));

        i += 1;
    }
}

#[inline(always)]
/// A generic vector subtraction implementation over one vector and single value.
///
/// # Safety
///
/// The sizes of `a`, `b` and `result` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_sub_vector<T, R, M, B>(
    dims: usize,
    a: &[T],
    b: &[T],
    mut result: &mut [B],
) where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
    for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
{
    debug_assert_eq!(a.len(), dims, "Vector a does not match size `dims`");
    debug_assert_eq!(b.len(), dims, "Vector a does not match size `dims`");
    debug_assert_eq!(
        result.raw_buffer_len(),
        dims,
        "Vector result does not match size `dims`"
    );

    let offset_from = dims % R::elements_per_dense();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let result_ptr = result.as_write_only_ptr();

    // Operate over dense lanes first.
    let mut i = 0;
    while i < (dims - offset_from) {
        let l1 = R::load_dense(a_ptr.add(i));
        let l2 = R::load_dense(b_ptr.add(i));
        let res = R::sub_dense(l1, l2);
        R::write_dense(result_ptr.add(i), res);

        i += R::elements_per_dense();
    }

    // Operate over single registers next.
    let offset_from = offset_from % R::elements_per_lane();
    while i < (dims - offset_from) {
        let l1 = R::load(a_ptr.add(i));
        let l2 = R::load(b_ptr.add(i));
        let res = R::sub(l1, l2);
        R::write(result_ptr.add(i), res);

        i += R::elements_per_lane();
    }

    while i < dims {
        let a = *a.get_unchecked(i);
        let b = *b.get_unchecked(i);
        result.write_at(i, M::sub(a, b));

        i += 1;
    }
}

#[inline(always)]
/// A generic vector multiplication implementation over one vector and single value.
///
/// # Safety
///
/// The sizes of `a`, `b` and `result` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_mul_vector<T, R, M, B>(
    dims: usize,
    a: &[T],
    b: &[T],
    mut result: &mut [B],
) where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
    for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
{
    debug_assert_eq!(a.len(), dims, "Vector a does not match size `dims`");
    debug_assert_eq!(b.len(), dims, "Vector a does not match size `dims`");
    debug_assert_eq!(
        result.raw_buffer_len(),
        dims,
        "Vector result does not match size `dims`"
    );

    let offset_from = dims % R::elements_per_dense();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let result_ptr = result.as_write_only_ptr();

    // Operate over dense lanes first.
    let mut i = 0;
    while i < (dims - offset_from) {
        let l1 = R::load_dense(a_ptr.add(i));
        let l2 = R::load_dense(b_ptr.add(i));
        let res = R::mul_dense(l1, l2);
        R::write_dense(result_ptr.add(i), res);

        i += R::elements_per_dense();
    }

    // Operate over single registers next.
    let offset_from = offset_from % R::elements_per_lane();
    while i < (dims - offset_from) {
        let l1 = R::load(a_ptr.add(i));
        let l2 = R::load(b_ptr.add(i));
        let res = R::mul(l1, l2);
        R::write(result_ptr.add(i), res);

        i += R::elements_per_lane();
    }

    while i < dims {
        let a = *a.get_unchecked(i);
        let b = *b.get_unchecked(i);
        result.write_at(i, M::mul(a, b));

        i += 1;
    }
}

#[inline(always)]
/// A generic vector division implementation dividing by vector `b`.
///
/// # Safety
///
/// The sizes of `a`, `b` and `result` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_div_vector<T, R, M, B>(
    dims: usize,
    a: &[T],
    b: &[T],
    mut result: &mut [B],
) where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
    for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
{
    debug_assert_eq!(a.len(), dims, "Vector a does not match size `dims`");
    debug_assert_eq!(b.len(), dims, "Vector a does not match size `dims`");
    debug_assert_eq!(
        result.raw_buffer_len(),
        dims,
        "Vector result does not match size `dims`"
    );

    let offset_from = dims % R::elements_per_dense();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let result_ptr = result.as_write_only_ptr();

    // Operate over dense lanes first.
    let mut i = 0;
    while i < (dims - offset_from) {
        let l1 = R::load_dense(a_ptr.add(i));
        let l2 = R::load_dense(b_ptr.add(i));
        let res = R::div_dense(l1, l2);
        R::write_dense(result_ptr.add(i), res);

        i += R::elements_per_dense();
    }

    // Operate over single registers next.
    let offset_from = offset_from % R::elements_per_lane();
    while i < (dims - offset_from) {
        let l1 = R::load(a_ptr.add(i));
        let l2 = R::load(b_ptr.add(i));
        let res = R::div(l1, l2);
        R::write(result_ptr.add(i), res);

        i += R::elements_per_lane();
    }

    while i < dims {
        let a = *a.get_unchecked(i);
        let b = *b.get_unchecked(i);
        result.write_at(i, M::div(a, b));

        i += 1;
    }
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
        generic_add_vector::<T, R, AutoMath, _>(dims, &l1, &l2, &mut result);

        let mut expected_result = Vec::new();
        for (a, b) in l1.iter().copied().zip(l2) {
            expected_result.push(AutoMath::add(a, b));
        }
        assert_eq!(result, expected_result, "value missmatch");
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
        generic_sub_vector::<T, R, AutoMath, _>(dims, &l1, &l2, &mut result);

        let mut expected_result = Vec::new();
        for (a, b) in l1.iter().copied().zip(l2) {
            expected_result.push(AutoMath::sub(a, b));
        }
        assert_eq!(result, expected_result, "value missmatch");
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
        generic_div_vector::<T, R, AutoMath, _>(dims, &l1, &l2, &mut result);

        let mut expected_result = Vec::new();
        for (a, b) in l1.iter().copied().zip(l2) {
            expected_result.push(AutoMath::div(a, b));
        }
        assert_eq!(result, expected_result, "value missmatch");
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
        generic_mul_vector::<T, R, AutoMath, _>(dims, &l1, &l2, &mut result);

        let mut expected_result = Vec::new();
        for (a, b) in l1.iter().copied().zip(l2) {
            expected_result.push(AutoMath::mul(a, b));
        }
        assert_eq!(result, expected_result, "value missmatch");
    }
}
