use crate::danger::core_simd_api::SimdRegister;
use crate::math::Math;

#[inline(always)]
/// A generic horizontal min implementation over one vectors of a given set of dimensions.
///
/// # Safety
///
/// The sizes of `a` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_min_horizontal<T, R, M>(dims: usize, a: &[T]) -> T
where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
{
    debug_assert_eq!(a.len(), dims, "Vector a does not match size `dims`");

    let offset_from = dims % R::elements_per_dense();
    let a_ptr = a.as_ptr();

    let mut min = R::filled_dense(M::max());

    // Operate over dense lanes first.
    let mut i = 0;
    while i < (dims - offset_from) {
        let l1 = R::load_dense(a_ptr.add(i));
        min = R::min_dense(min, l1);

        i += R::elements_per_dense();
    }

    let mut min = R::min_to_register(min);

    // Operate over single registers next.
    let offset_from = offset_from % R::elements_per_lane();
    while i < (dims - offset_from) {
        let l1 = R::load(a_ptr.add(i));
        min = R::min(min, l1);

        i += R::elements_per_lane();
    }

    // Handle the remainder.
    let mut min = R::min_to_value(min);

    while i < dims {
        let a = *a.get_unchecked(i);
        min = M::cmp_min(min, a);

        i += 1;
    }

    min
}

// TODO: Are we really sure this is the most optimal way of doing vertical max/min/etc...
//       or should we be stepping through the entire matrix each time? Probably this since
//       I imagine it is more friendly on the cache. But we should test 0-0
#[inline(always)]
/// A generic vertical min implementation over two vectors of a given set of dimensions.
///
/// NOTE:
/// This implementation with compared the values of `a` and `b` and store the min
/// of the each element in `b`.
///
/// # Safety
///
/// The sizes of `a`, `b` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_min_vertical<T, R, M>(
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
        let min = R::min_dense(l1, l2);
        R::write_dense(result_ptr.add(i), min);

        i += R::elements_per_dense();
    }

    // Operate over single registers next.
    let offset_from = offset_from % R::elements_per_lane();
    while i < (dims - offset_from) {
        let l1 = R::load(a_ptr.add(i));
        let l2 = R::load(b_ptr.add(i));
        let min = R::min(l1, l2);
        R::write(result_ptr.add(i), min);

        i += R::elements_per_lane();
    }

    while i < dims {
        let v1 = *a.get_unchecked(i);
        let v2 = *b.get_unchecked(i);
        *result.get_unchecked_mut(i) = M::cmp_min(v1, v2);

        i += 1;
    }
}

#[inline(always)]
/// A generic min implementation over one vector of a given set of dimensions
/// and a single value target.
///
/// # Safety
///
/// The sizes of `a` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_min_value<T, R, M>(
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
        let min = R::min_dense(l1, broadcast_dense);
        R::write_dense(result_ptr.add(i), min);

        i += R::elements_per_dense();
    }

    let broadcast_reg = broadcast_dense.a;

    // Operate over single registers next.
    let offset_from = offset_from % R::elements_per_lane();
    while i < (dims - offset_from) {
        let l1 = R::load(a_ptr.add(i));
        let min = R::min(l1, broadcast_reg);
        R::write(result_ptr.add(i), min);

        i += R::elements_per_lane();
    }

    while i < dims {
        let v1 = *a.get_unchecked(i);
        *result.get_unchecked_mut(i) = M::cmp_min(v1, value);

        i += 1;
    }
}

#[cfg(test)]
pub(crate) unsafe fn test_min<T, R>(l1: Vec<T>, l2: Vec<T>)
where
    T: Copy + PartialEq + std::fmt::Debug,
    R: SimdRegister<T>,
    crate::math::AutoMath: Math<T>,
{
    use crate::math::AutoMath;

    let dims = l1.len();
    let mut result = vec![AutoMath::max(); dims];
    generic_min_vertical::<T, R, AutoMath>(dims, &l1, &l2, &mut result);

    let mut expected_result = Vec::new();
    for (a, b) in l1.iter().copied().zip(l2) {
        expected_result.push(AutoMath::cmp_min(a, b));
    }
    assert_eq!(result, expected_result, "value mismatch");

    let dims = l1.len();
    let mut result = vec![AutoMath::max(); dims];
    generic_min_value::<T, R, AutoMath>(dims, AutoMath::zero(), &l1, &mut result);
    let mut expected_result = Vec::new();
    for a in l1.iter().copied() {
        expected_result.push(AutoMath::cmp_min(a, AutoMath::zero()));
    }
    assert_eq!(result, expected_result, "value mismatch");
    
    let min = generic_min_horizontal::<T, R, AutoMath>(dims, &l1);
    let expected_min = l1
        .iter()
        .fold(AutoMath::max(), |a, b| AutoMath::cmp_min(a, *b));
    assert_eq!(min, expected_min, "value mismatch on horizontal");
}
