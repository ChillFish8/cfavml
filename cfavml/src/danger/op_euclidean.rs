use crate::danger::core_simd_api::SimdRegister;
use crate::math::Math;
use crate::mem_loader::{IntoMemLoader, MemLoader};

#[inline(always)]
/// A generic squared Euclidean distance implementation over two vectors of a given set of dimensions.
///
/// # Safety
///
/// The sizes of `a` and `b` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_squared_euclidean<T, R, M, B1, B2>(a: B1, b: B2) -> T
where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
    B1: IntoMemLoader<T>,
    B1::Loader: MemLoader<Value = T>,
    B2: IntoMemLoader<T>,
    B2::Loader: MemLoader<Value = T>,
{
    let mut a = a.into_mem_loader();
    let mut b = b.into_mem_loader();
    assert_eq!(
        a.projected_len(),
        b.projected_len(),
        "Buffers `a` and `b` do not match in size"
    );

    let len = a.projected_len();
    let offset_from = len % R::elements_per_dense();

    let mut total = R::zeroed_dense();

    // Operate over dense lanes first.
    let mut i = 0;
    while i < (len - offset_from) {
        let l1 = a.load_dense::<R>();
        let l2 = b.load_dense::<R>();
        let diff = R::sub_dense(l1, l2);
        total = R::fmadd_dense(diff, diff, total);

        i += R::elements_per_dense();
    }

    let mut total = R::sum_to_register(total);

    // Operate over single registers next.
    let offset_from = offset_from % R::elements_per_lane();
    while i < (len - offset_from) {
        let l1 = a.load::<R>();
        let l2 = b.load::<R>();
        let diff = R::sub(l1, l2);
        total = R::fmadd(diff, diff, total);

        i += R::elements_per_lane();
    }

    // Handle the remainder.
    let mut total = R::sum_to_value(total);

    while i < len {
        let a = a.read();
        let b = b.read();
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

    let value = generic_squared_euclidean::<T, R, AutoMath, _, _>(&l1, &l2);
    let expected_value = crate::test_utils::simple_euclidean(&l1, &l2);
    assert!(
        AutoMath::is_close(value, expected_value),
        "value missmatch {value:?} vs {expected_value:?}"
    );
}
