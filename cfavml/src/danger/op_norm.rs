use crate::danger::core_simd_api::SimdRegister;
use crate::math::Math;
use crate::mem_loader::{IntoMemLoader, MemLoader};

#[inline(always)]
/// A generic squared norm implementation over a vectors of a given set of dimensions.
///
/// # Safety
///
/// The sizes of `a` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_squared_norm<T, R, M, B1>(a: B1) -> T
where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
    B1: IntoMemLoader<T>,
    B1::Loader: MemLoader<Value = T>,
{
    let mut a = a.into_mem_loader();

    let len = a.projected_len();
    let offset_from = len % R::elements_per_dense();

    let mut total = R::zeroed_dense();

    // Operate over dense lanes first.
    let mut i = 0;
    while i < (len - offset_from) {
        let l1 = a.load_dense::<R>();
        total = R::fmadd_dense(l1, l1, total);

        i += R::elements_per_dense();
    }

    let mut total = R::sum_to_register(total);

    // Operate over single registers next.
    let offset_from = offset_from % R::elements_per_lane();
    while i < (len - offset_from) {
        let l1 = a.load::<R>();
        total = R::fmadd(l1, l1, total);

        i += R::elements_per_lane();
    }

    // Handle the remainder.
    let mut total = R::sum_to_value(total);

    while i < len {
        let a = a.read();
        total = M::add(total, M::mul(a, a));

        i += 1;
    }

    total
}

#[cfg(test)]
pub(crate) unsafe fn test_squared_norm<T, R>(l1: Vec<T>)
where
    T: Copy + PartialEq + std::fmt::Debug,
    R: SimdRegister<T>,
    crate::math::AutoMath: Math<T>,
{
    use crate::math::AutoMath;

    let value = generic_squared_norm::<T, R, AutoMath, _>(&l1);
    let expected_value = crate::test_utils::simple_dot(&l1, &l1);
    assert!(
        AutoMath::is_close(value, expected_value),
        "value missmatch {value:?} vs {expected_value:?}"
    );
}
