use crate::danger::core_simd_api::SimdRegister;
use crate::math::Math;

#[inline(always)]
/// A generic cosine implementation over two vectors of a given set of dimensions.
///
/// # Safety
///
/// The sizes of `a` and `b` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_cosine<T, R, M>(dims: usize, a: &[T], b: &[T]) -> T
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

    let mut norm_a = R::zeroed_dense();
    let mut norm_b = R::zeroed_dense();

    // Operate over dense lanes first.
    let mut i = 0;
    while i < (dims - offset_from) {
        let l1 = R::load_dense(a_ptr.add(i));
        let l2 = R::load_dense(b_ptr.add(i));

        norm_a = R::fmadd_dense(l1, l1, norm_a);
        norm_b = R::fmadd_dense(l2, l2, norm_b);

        i += R::elements_per_dense();
    }

    let mut norm_a = R::sum_to_register(norm_a);
    let mut norm_b = R::sum_to_register(norm_b);

    // Operate over single registers next.
    let offset_from = offset_from % R::elements_per_lane();
    while i < (dims - offset_from) {
        let l1 = R::load(a_ptr.add(i));
        let l2 = R::load(b_ptr.add(i));
        norm_a = R::fmadd(l1, l1, norm_a);
        norm_b = R::fmadd(l2, l2, norm_b);

        i += R::elements_per_lane();
    }

    // Handle the remainder.
    let mut norm_a = R::sum_to_value(norm_a);
    let mut norm_b = R::sum_to_value(norm_b);

    while i < dims {
        let a = *a.get_unchecked(i);
        let b = *b.get_unchecked(i);
        norm_a = M::add(norm_a, M::mul(a, a));
        norm_b = M::add(norm_b, M::mul(b, b));

        i += 1;
    }

    let dot = super::op_dot_product::generic_dot_product::<T, R, M>(dims, a, b);
    cosine::<T, M>(dot, norm_a, norm_b)
}

#[inline(always)]
pub(crate) fn cosine<T: Copy, M: Math<T>>(dot_product: T, norm_x: T, norm_y: T) -> T {
    if M::cmp_eq(norm_x, M::zero()) && M::cmp_eq(norm_y, M::zero()) {
        M::zero()
    } else if M::cmp_eq(norm_x, M::zero()) || M::cmp_eq(norm_y, M::zero()) {
        M::one()
    } else {
        M::sub(
            M::one(),
            M::div(dot_product, M::sqrt(M::mul(norm_x, norm_y))),
        )
    }
}

#[cfg(test)]
pub(crate) unsafe fn test_cosine<T, R>(l1: Vec<T>, l2: Vec<T>)
where
    T: Copy + PartialEq + std::fmt::Debug,
    R: SimdRegister<T>,
    crate::math::AutoMath: Math<T>,
{
    use crate::math::AutoMath;

    let dims = l1.len();
    let value = generic_cosine::<T, R, AutoMath>(dims, &l1, &l2);
    let expected_value = crate::test_utils::simple_cosine(&l1, &l2);
    assert!(
        AutoMath::is_close(value, expected_value),
        "value missmatch {value:?} vs {expected_value:?}"
    );
}
