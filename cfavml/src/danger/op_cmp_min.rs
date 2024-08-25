use crate::buffer::WriteOnlyBuffer;
use crate::danger::core_routine_boilerplate::apply_vertical_kernel;
use crate::danger::core_simd_api::SimdRegister;
use crate::math::Math;
use crate::mem_loader::{IntoMemLoader, MemLoader};

#[inline(always)]
/// A generic horizontal min implementation over one vectors of a given set of dimensions.
///
/// # Safety
///
/// The sizes of `a` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_cmp_min<T, R, M, B1>(a: B1) -> T
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

    let mut min = R::filled_dense(M::max());

    // Operate over dense lanes first.
    let mut i = 0;
    while i < (len - offset_from) {
        let l1 = a.load_dense::<R>();
        min = R::min_dense(min, l1);

        i += R::elements_per_dense();
    }

    let mut min = R::min_to_register(min);

    // Operate over single registers next.
    let offset_from = offset_from % R::elements_per_lane();
    while i < (len - offset_from) {
        let l1 = a.load::<R>();
        min = R::min(min, l1);

        i += R::elements_per_lane();
    }

    // Handle the remainder.
    let mut min = R::min_to_value(min);

    while i < len {
        min = M::cmp_min(min, a.read());

        i += 1;
    }

    min
}

#[inline(always)]
/// A generic vertical min implementation over two vectors of a given set of dimensions.
///
/// NOTE:
/// This implementation with compared the values of `a` and `b` and store the min
/// of the two elements in `result`.
///
/// # Safety
///
/// The sizes of `a`, `b` and `result` must be equal to `dims`, the safety requirements of
/// `M` definition the basic math operations and the requirements of `R` SIMD register
/// must also be followed.
pub unsafe fn generic_cmp_min_vertical<T, R, M, B1, B2, B3>(
    a: B1,
    b: B2,
    result: &mut [B3],
) where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
    B1: IntoMemLoader<T>,
    B1::Loader: MemLoader<Value = T>,
    B2: IntoMemLoader<T>,
    B2::Loader: MemLoader<Value = T>,
    for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = T>,
{
    apply_vertical_kernel::<T, R, M, B1, B2, B3>(
        a,
        b,
        result,
        R::min_dense,
        R::min,
        M::cmp_min,
    );
}

#[cfg(test)]
pub(crate) unsafe fn test_min<T, R>(l1: Vec<T>, l2: Vec<T>)
where
    T: Copy + PartialEq + std::fmt::Debug + IntoMemLoader<T>,
    R: SimdRegister<T>,
    crate::math::AutoMath: Math<T>,
    for<'a> &'a Vec<T>: IntoMemLoader<T>,
    for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
{
    use crate::math::AutoMath;

    let dims = l1.len();
    let mut result = vec![AutoMath::max(); dims];
    generic_cmp_min_vertical::<T, R, AutoMath, _, _, _>(&l1, &l2, &mut result);

    let mut expected_result = Vec::new();
    for (a, b) in l1.iter().copied().zip(l2) {
        expected_result.push(AutoMath::cmp_min(a, b));
    }
    assert_eq!(result, expected_result, "value mismatch");

    let dims = l1.len();
    let mut result = vec![AutoMath::max(); dims];
    generic_cmp_min_vertical::<T, R, AutoMath, _, _, _>(
        AutoMath::zero(),
        &l1,
        &mut result,
    );
    let mut expected_result = Vec::new();
    for a in l1.iter().copied() {
        expected_result.push(AutoMath::cmp_min(a, AutoMath::zero()));
    }
    assert_eq!(result, expected_result, "value mismatch");

    let min = generic_cmp_min::<T, R, AutoMath, _>(&l1);
    let expected_min = l1
        .iter()
        .fold(AutoMath::max(), |a, b| AutoMath::cmp_min(a, *b));
    assert_eq!(min, expected_min, "value mismatch on horizontal");
}
