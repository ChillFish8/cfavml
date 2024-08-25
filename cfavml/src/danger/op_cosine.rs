use crate::danger::core_simd_api::SimdRegister;
use crate::math::Math;
use crate::mem_loader::{IntoMemLoader, MemLoader};

#[inline(always)]
/// A generic cosine implementation over two vectors of a given set of dimensions.
///
/// # Panics
///
/// If `a` and `b` are not the same length; no projection is available on this routine.
///
/// # Safety
///
/// The safety requirements of `M` definition the basic math operations and
/// the requirements of `R` SIMD register must also be followed.
pub unsafe fn generic_cosine<T, R, M, B1, B2>(a: B1, b: B2) -> T
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
    let offset_from = len % R::elements_per_lane();

    let mut norm_a = R::zeroed();
    let mut norm_b = R::zeroed();
    let mut dot = R::zeroed();

    // Operate over single registers, cosine puts too much pressure on registers
    // on AVX2 to support doing this via dense lanes. Hopefully the compiler slightly
    // unrolls this loop so we don't pay as much for branching.
    let mut i = 0;
    while i < (len - offset_from) {
        let l1 = a.load::<R>();
        let l2 = b.load::<R>();

        norm_a = R::fmadd(l1, l1, norm_a);
        norm_b = R::fmadd(l2, l2, norm_b);
        dot = R::fmadd(l1, l2, dot);

        i += R::elements_per_lane();
    }

    // Handle the remainder.
    let mut norm_a = R::sum_to_value(norm_a);
    let mut norm_b = R::sum_to_value(norm_b);
    let mut dot = R::sum_to_value(dot);

    while i < len {
        let a = a.read();
        let b = b.read();
        norm_a = M::add(norm_a, M::mul(a, a));
        norm_b = M::add(norm_b, M::mul(b, b));
        dot = M::add(dot, M::mul(a, b));

        i += 1;
    }

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

    let value = generic_cosine::<T, R, AutoMath, _, _>(&l1, &l2);
    let expected_value = crate::test_utils::simple_cosine(&l1, &l2);
    assert!(
        AutoMath::is_close(value, expected_value),
        "value missmatch {value:?} vs {expected_value:?}"
    );
}
