use crate::danger::core_simd_api::SimdRegister;
use crate::math::Math;

#[inline(always)]
/// A generic Euclidean distance implementation over two vectors of a given set of dimensions.
///
/// # Safety
///
/// The sizes of `a` and `b` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_euclidean<T, R, M>(dims: usize, a: &[T], b: &[T]) -> T
where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
{
    debug_assert_eq!(a.len(), dims, "Vector a does not match size `dims`");
    debug_assert_eq!(b.len(), dims, "Vector b does not match size `dims`");

    let offset_from = dims % R::elements_per_dense();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut total = R::zeroed_dense();

    // Operate over dense lanes first.
    let mut i = 0;
    while i < (dims - offset_from) {
        let l1 = R::load_dense(a_ptr.add(i));
        let l2 = R::load_dense(b_ptr.add(i));
        let diff = R::sub_dense(l1, l2);
        total = R::fmadd_dense(diff, diff, total);

        i += R::elements_per_dense();
    }

    let mut total = R::sum_to_register(total);

    // Operate over single registers next.
    let offset_from = offset_from % R::elements_per_lane();
    while i < (dims - offset_from) {
        let l1 = R::load(a_ptr.add(i));
        let l2 = R::load(b_ptr.add(i));
        let diff = R::sub(l1, l2);
        total = R::fmadd(diff, diff, total);

        i += R::elements_per_lane();
    }

    // Handle the remainder.
    let mut total = R::sum_to_value(total);

    while i < dims {
        let a = *a.get_unchecked(i);
        let b = *b.get_unchecked(i);
        let diff = M::sub(a, b);
        total = M::add(total, M::mul(diff, diff));

        i += 1;
    }

    total
}

#[cfg(test)]
pub(crate) unsafe fn test_euclidean<T, R>(l1: Vec<T>, l2: Vec<T>)
where
    T: Copy + PartialEq + std::fmt::Debug,
    R: SimdRegister<T>,
    crate::math::AutoMath: Math<T>,
{
    use crate::math::AutoMath;

    let dims = l1.len();
    let value = generic_euclidean::<T, R, AutoMath>(dims, &l1, &l2);
    let expected_value = crate::test_utils::simple_euclidean(&l1, &l2);
    assert!(
        AutoMath::is_close(value, expected_value),
        "value missmatch {value:?} vs {expected_value:?}"
    );
}
