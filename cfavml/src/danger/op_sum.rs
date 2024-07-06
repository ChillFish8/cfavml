use crate::danger::core_simd_api::SimdRegister;
use crate::math::Math;

#[inline(always)]
/// A generic horizontal sum implementation over one vectors of a given set of dimensions.
///
/// # Safety
///
/// The sizes of `a` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_sum_horizontal<T, R, M>(dims: usize, a: &[T]) -> T
where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
{
    debug_assert_eq!(a.len(), dims, "Vector a does not match size `dims`");

    let offset_from = dims % R::elements_per_dense();
    let a_ptr = a.as_ptr();

    let mut sum = R::zeroed_dense();

    // Operate over dense lanes first.
    let mut i = 0;
    while i < (dims - offset_from) {
        let l1 = R::load_dense(a_ptr.add(i));
        sum = R::add_dense(sum, l1);

        i += R::elements_per_dense();
    }

    let mut sum = R::sum_to_register(sum);

    // Operate over single registers next.
    let offset_from = offset_from % R::elements_per_lane();
    while i < (dims - offset_from) {
        let l1 = R::load(a_ptr.add(i));
        sum = R::add(sum, l1);

        i += R::elements_per_lane();
    }

    // Handle the remainder.
    let mut sum = R::sum_to_value(sum);

    while i < dims {
        let a = *a.get_unchecked(i);
        sum = M::add(sum, a);

        i += 1;
    }

    sum
}

// TODO: Are we really sure this is the most optimal way of doing vertical max/min/etc...
//       or should we be stepping through the entire matrix each time? Probably this since
//       I imagine it is more friendly on the cache. But we should test 0-0
#[inline(always)]
/// A generic vertical sum implementation over two vectors of a given set of dimensions.
///
/// NOTE:
/// This implementation with compared the values of `a` and `b` and store the sum
/// of the each element in `b`.
///
/// # Safety
///
/// The sizes of `a`, `b` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_sum_vertical<T, R, M>(
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
        let sum = R::add_dense(l1, l2);
        R::write_dense(result_ptr.add(i), sum);

        i += R::elements_per_dense();
    }

    // Operate over single registers next.
    let offset_from = offset_from % R::elements_per_lane();
    while i < (dims - offset_from) {
        let l1 = R::load(a_ptr.add(i));
        let l2 = R::load(b_ptr.add(i));
        let sum = R::add(l1, l2);
        R::write(result_ptr.add(i), sum);

        i += R::elements_per_lane();
    }

    while i < dims {
        let a = *a.get_unchecked(i);
        let b = *b.get_unchecked(i);
        *result.get_unchecked_mut(i) = M::add(a, b);

        i += 1;
    }
}

#[cfg(test)]
pub(crate) unsafe fn test_sum<T, R>(l1: Vec<T>, l2: Vec<T>)
where
    T: Copy + PartialEq + std::fmt::Debug,
    R: SimdRegister<T>,
    crate::math::AutoMath: Math<T>,
{
    use crate::math::AutoMath;

    let dims = l1.len();
    let mut result = vec![AutoMath::zero(); dims];
    generic_sum_vertical::<T, R, AutoMath>(dims, &l1, &l2, &mut result);

    let mut expected_result = Vec::new();
    for (a, b) in l1.iter().copied().zip(l2) {
        expected_result.push(AutoMath::add(a, b));
    }
    assert_eq!(result, expected_result, "value missmatch");

    let sum = generic_sum_horizontal::<T, R, AutoMath>(dims, &l1);
    let expected_sum = l1
        .iter()
        .fold(AutoMath::zero(), |a, b| AutoMath::add(a, *b));
    assert!(
        AutoMath::is_close(sum, expected_sum),
        "value missmatch on horizontal {sum:?} vs {expected_sum:?}"
    );
}
