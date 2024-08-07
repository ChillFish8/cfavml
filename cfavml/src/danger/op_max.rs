use crate::danger::core_simd_api::SimdRegister;
use crate::math::Math;

#[inline(always)]
/// A generic horizontal max implementation over one vectors of a given set of dimensions.
///
/// # Safety
///
/// The sizes of `a` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_max_horizontal<T, R, M>(dims: usize, a: &[T]) -> T
where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
{
    debug_assert_eq!(a.len(), dims, "Vector a does not match size `dims`");

    let offset_from = dims % R::elements_per_dense();
    let a_ptr = a.as_ptr();

    let mut max = R::filled_dense(M::min());

    // Operate over dense lanes first.
    let mut i = 0;
    while i < (dims - offset_from) {
        let l1 = R::load_dense(a_ptr.add(i));
        max = R::max_dense(max, l1);

        i += R::elements_per_dense();
    }

    let mut max = R::max_to_register(max);

    // Operate over single registers next.
    let offset_from = offset_from % R::elements_per_lane();
    while i < (dims - offset_from) {
        let l1 = R::load(a_ptr.add(i));
        max = R::max(max, l1);

        i += R::elements_per_lane();
    }

    // Handle the remainder.
    let mut max = R::max_to_value(max);

    while i < dims {
        let a = *a.get_unchecked(i);
        max = M::cmp_max(max, a);

        i += 1;
    }

    max
}

// TODO: Are we really sure this is the most optimal way of doing vertical max/min/etc...
//       or should we be stepping through the entire matrix each time? Probably this since
//       I imagine it is more friendly on the cache. But we should test 0-0
#[inline(always)]
/// A generic vertical max implementation over two vectors of a given set of dimensions.
///
/// NOTE:
/// This implementation with compared the values of `a` and `b` and store the max
/// of the element in `b`.
///
/// # Safety
///
/// The sizes of `a`, `b` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_max_vertical<T, R, M>(
    dims: usize,
    a: &[T],
    b: &[T],
    result: &mut [T],
) where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
{
    debug_assert_eq!(a.len(), dims, "Vector a does not match size `dims`");
    debug_assert_eq!(b.len(), dims, "Vector result does not match size `dims`");

    let offset_from = dims % R::elements_per_dense();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let result_ptr = result.as_mut_ptr();

    // Operate over dense lanes first.
    let mut i = 0;
    while i < (dims - offset_from) {
        let l1 = R::load_dense(a_ptr.add(i));
        let l2 = R::load_dense(b_ptr.add(i));
        let max = R::max_dense(l1, l2);
        R::write_dense(result_ptr.add(i), max);

        i += R::elements_per_dense();
    }

    // Operate over single registers next.
    let offset_from = offset_from % R::elements_per_lane();
    while i < (dims - offset_from) {
        let l1 = R::load(a_ptr.add(i));
        let l2 = R::load(b_ptr.add(i));
        let max = R::max(l1, l2);
        R::write(result_ptr.add(i), max);

        i += R::elements_per_lane();
    }

    while i < dims {
        let v1 = *a.get_unchecked(i);
        let v2 = *b.get_unchecked(i);
        *result.get_unchecked_mut(i) = M::cmp_max(v1, v2);

        i += 1;
    }
}

#[inline(always)]
/// A generic max implementation over one vector of a given set of dimensions
/// and a single value target.
///
/// This routine is primarily aimed for workloads like Relu activation where
/// you want to cap the value at zero or above.
///
/// # Safety
///
/// The sizes of `a` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_max_value<T, R, M>(
    dims: usize,
    value: T,
    a: &[T],
    result: &mut [T],
) where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
{
    debug_assert_eq!(a.len(), dims, "Vector a does not match size `dims`");

    let broadcast_dense = R::filled_dense(value);

    let offset_from = dims % R::elements_per_dense();
    let a_ptr = a.as_ptr();
    let result_ptr = result.as_mut_ptr();

    // Operate over dense lanes first.
    let mut i = 0;
    while i < (dims - offset_from) {
        let l1 = R::load_dense(a_ptr.add(i));
        let max = R::max_dense(l1, broadcast_dense);
        R::write_dense(result_ptr.add(i), max);

        i += R::elements_per_dense();
    }

    let broadcast_reg = broadcast_dense.a;

    // Operate over single registers next.
    let offset_from = offset_from % R::elements_per_lane();
    while i < (dims - offset_from) {
        let l1 = R::load(a_ptr.add(i));
        let max = R::max(l1, broadcast_reg);
        R::write(result_ptr.add(i), max);

        i += R::elements_per_lane();
    }

    while i < dims {
        let v1 = *a.get_unchecked(i);
        *result.get_unchecked_mut(i) = M::cmp_max(v1, value);

        i += 1;
    }
}

#[cfg(test)]
pub(crate) unsafe fn test_max<T, R>(l1: Vec<T>, l2: Vec<T>)
where
    T: Copy + PartialEq + std::fmt::Debug,
    R: SimdRegister<T>,
    crate::math::AutoMath: Math<T>,
{
    use crate::math::AutoMath;

    let dims = l1.len();
    let mut result = vec![AutoMath::min(); dims];
    generic_max_vertical::<T, R, AutoMath>(dims, &l1, &l2, &mut result);
    let mut expected_result = Vec::new();
    for (a, b) in l1.iter().copied().zip(l2) {
        expected_result.push(AutoMath::cmp_max(a, b));
    }
    assert_eq!(result, expected_result, "value mismatch");

    let dims = l1.len();
    let mut result = vec![AutoMath::min(); dims];
    generic_max_value::<T, R, AutoMath>(dims, AutoMath::zero(), &l1, &mut result);
    let mut expected_result = Vec::new();
    for a in l1.iter().copied() {
        expected_result.push(AutoMath::cmp_max(a, AutoMath::zero()));
    }
    assert_eq!(result, expected_result, "value mismatch");

    let max = generic_max_horizontal::<T, R, AutoMath>(dims, &l1);
    let expected_max = l1
        .iter()
        .fold(AutoMath::min(), |a, b| AutoMath::cmp_max(a, *b));
    assert_eq!(max, expected_max, "value mismatch on horizontal");
}
