use crate::danger::core_simd_api::SimdRegister;
use crate::math::Math;
use crate::mem_loader::{IntoMemLoader, MemLoader};

#[inline(always)]
/// A generic horizontal sum implementation over one vectors of a given set of dimensions.
///
/// # Safety
///
/// The sizes of `a` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_sum<T, R, M, B1>(a: B1) -> T
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

    let mut sum = R::zeroed_dense();

    // Operate over dense lanes first.
    let mut i = 0;
    while i < (len - offset_from) {
        let l1 = a.load_dense::<R>();
        sum = R::add_dense(sum, l1);

        i += R::elements_per_dense();
    }

    let mut sum = R::sum_to_register(sum);

    // Operate over single registers next.
    let offset_from = offset_from % R::elements_per_lane();
    while i < (len - offset_from) {
        let l1 = a.load::<R>();
        sum = R::add(sum, l1);

        i += R::elements_per_lane();
    }

    // Handle the remainder.
    let mut sum = R::sum_to_value(sum);

    while i < len {
        sum = M::add(sum, a.read());

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

    let sum = generic_sum::<T, R, AutoMath, _>(&l1);
    let expected_sum = l1
        .iter()
        .fold(AutoMath::zero(), |a, b| AutoMath::add(a, *b));
    assert!(
        AutoMath::is_close(sum, expected_sum),
        "value missmatch on horizontal {sum:?} vs {expected_sum:?}"
    );
}
