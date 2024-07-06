use crate::danger::core_simd_api::SimdRegister;
use crate::math::Math;

#[inline(always)]
/// A generic squared norm implementation over a vectors of a given set of dimensions.
///
/// # Safety
///
/// The sizes of `a` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_squared_norm<T, R, M>(dims: usize, a: &[T]) -> T
where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
{
    debug_assert_eq!(a.len(), dims, "Vector a does not match size `dims`");

    let offset_from = dims % R::elements_per_dense();
    let a_ptr = a.as_ptr();

    let mut total = R::zeroed_dense();

    // Operate over dense lanes first.
    let mut i = 0;
    while i < (dims - offset_from) {
        let l1 = R::load_dense(a_ptr.add(i));
        total = R::fmadd_dense(l1, l1, total);

        i += R::elements_per_dense();
    }

    let mut total = R::sum_to_register(total);

    // Operate over single registers next.
    let offset_from = offset_from % R::elements_per_lane();
    while i < (dims - offset_from) {
        let l1 = R::load(a_ptr.add(i));
        total = R::fmadd(l1, l1, total);

        i += R::elements_per_lane();
    }

    // Handle the remainder.
    let mut total = R::sum_to_value(total);

    while i < dims {
        let a = *a.get_unchecked(i);
        total = M::add(total, M::mul(a, a));

        i += 1;
    }

    total
}
