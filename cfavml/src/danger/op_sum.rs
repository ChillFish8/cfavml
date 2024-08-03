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

#[cfg(test)]
pub(crate) unsafe fn test_sum<T, R>(l1: Vec<T>)
where
    T: Copy + PartialEq + std::fmt::Debug,
    R: SimdRegister<T>,
    crate::math::AutoMath: Math<T>,
{
    use crate::math::AutoMath;

    let dims = l1.len();

    let sum = generic_sum_horizontal::<T, R, AutoMath>(dims, &l1);
    let expected_sum = l1
        .iter()
        .fold(AutoMath::zero(), |a, b| AutoMath::add(a, *b));
    assert!(
        AutoMath::is_close(sum, expected_sum),
        "value missmatch on horizontal {sum:?} vs {expected_sum:?}"
    );
}
