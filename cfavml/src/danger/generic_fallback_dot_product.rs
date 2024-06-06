use crate::danger::utils::rollup_scalar_x8;
use crate::math::*;

#[inline]
/// Computes the dot product of two `T` vectors.
///
/// # Safety
///
/// Vectors **MUST** be equal in length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
pub unsafe fn generic_xany_fallback_nofma_dot<T>(x: &[T], y: &[T]) -> T
where
    T: Copy,
    AutoMath: Math<T>,
{
    fallback_dot::<T, AutoMath>(x, y)
}

#[inline]
pub(super) unsafe fn fallback_dot<T, M>(x: &[T], y: &[T]) -> T
where
    T: Copy,
    M: Math<T>,
{
    debug_assert_eq!(
        y.len(),
        x.len(),
        "Improper implementation detected, vectors must be equal length"
    );

    let len = x.len();
    let offset_from = len % 8;

    // We do this manual unrolling to allow the compiler to vectorize
    // the loop and avoid some branching even if we're not doing it explicitly.
    // This made a significant difference in benchmarking ~8x
    let mut acc1 = M::zero();
    let mut acc2 = M::zero();
    let mut acc3 = M::zero();
    let mut acc4 = M::zero();
    let mut acc5 = M::zero();
    let mut acc6 = M::zero();
    let mut acc7 = M::zero();
    let mut acc8 = M::zero();

    let mut i = 0;
    while i < offset_from {
        let x = *x.get_unchecked(i);
        let y = *y.get_unchecked(i);
        acc1 = M::add(acc1, M::mul(x, y));

        i += 1;
    }

    while i < len {
        let x1 = *x.get_unchecked(i);
        let x2 = *x.get_unchecked(i + 1);
        let x3 = *x.get_unchecked(i + 2);
        let x4 = *x.get_unchecked(i + 3);
        let x5 = *x.get_unchecked(i + 4);
        let x6 = *x.get_unchecked(i + 5);
        let x7 = *x.get_unchecked(i + 6);
        let x8 = *x.get_unchecked(i + 7);

        let y1 = *y.get_unchecked(i);
        let y2 = *y.get_unchecked(i + 1);
        let y3 = *y.get_unchecked(i + 2);
        let y4 = *y.get_unchecked(i + 3);
        let y5 = *y.get_unchecked(i + 4);
        let y6 = *y.get_unchecked(i + 5);
        let y7 = *y.get_unchecked(i + 6);
        let y8 = *y.get_unchecked(i + 7);

        acc1 = M::add(acc1, M::mul(x1, y1));
        acc2 = M::add(acc2, M::mul(x2, y2));
        acc3 = M::add(acc3, M::mul(x3, y3));
        acc4 = M::add(acc4, M::mul(x4, y4));
        acc5 = M::add(acc5, M::mul(x5, y5));
        acc6 = M::add(acc6, M::mul(x6, y6));
        acc7 = M::add(acc7, M::mul(x7, y7));
        acc8 = M::add(acc8, M::mul(x8, y8));

        i += 8;
    }

    rollup_scalar_x8::<T, M>(acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{assert_is_close, get_sample_vectors, simple_dot};

    #[test]
    fn test_f32_x1024_nofma_dot() {
        let (x, y) = get_sample_vectors(1024);
        let dist = unsafe { generic_xany_fallback_nofma_dot(&x, &y) };
        let expected = simple_dot(&x, &y);
        assert_is_close(dist, expected);
    }

    #[test]
    fn test_f32_xany_nofma_dot() {
        let (x, y) = get_sample_vectors(514);
        let dist = unsafe { generic_xany_fallback_nofma_dot(&x, &y) };
        let expected = simple_dot(&x, &y);
        assert_is_close(dist, expected);
    }
}
