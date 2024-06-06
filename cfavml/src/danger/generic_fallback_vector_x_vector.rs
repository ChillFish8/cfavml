use crate::math::*;

#[inline]
/// Divides each the input mutable vector `x` by the elements provided by `y`
/// element wise.
///
/// ```py
/// D: int
/// x: [T; D]
/// y: [T; D]
///
/// for i in 0..D:
///     x[i] = x[i] / y[i]
/// ```
///
/// # Safety
///
/// Lengths of `x` and `y` **MUST** be equal.
pub unsafe fn generic_xany_fallback_nofma_div_vertical<T>(x: &mut [T], y: &[T])
where
    T: Copy,
    AutoMath: Math<T>,
{
    f32_op_vertical(AutoMath::div, x, y)
}

#[inline]
/// Multiplies each the input mutable vector `x` by the elements provided by `y`
/// element wise.
///
/// ```py
/// D: int
/// x: [T; D]
/// y: [T; D]
///
/// for i in 0..D:
///     x[i] = x[i] * y[i]
/// ```
///
/// # Safety
///
/// Lengths of `x` and `y` **MUST** be equal.
pub unsafe fn generic_xany_fallback_nofma_mul_vertical<T>(x: &mut [T], y: &[T])
where
    T: Copy,
    AutoMath: Math<T>,
{
    f32_op_vertical(AutoMath::mul, x, y)
}

#[inline]
/// Adds each the input mutable vector `x` with the elements provided by `y`
/// element wise.
///
/// ```py
/// D: int
/// x: [T; D]
/// y: [T; D]
///
/// for i in 0..D:
///     x[i] = x[i] + y[i]
/// ```
///
/// # Safety
///
/// Lengths of `x` and `y` **MUST** be equal.
pub unsafe fn generic_xany_fallback_nofma_add_vertical<T>(x: &mut [T], y: &[T])
where
    T: Copy,
    AutoMath: Math<T>,
{
    f32_op_vertical(AutoMath::add, x, y)
}

#[inline]
/// Subtracts each the input mutable vector `x` by the elements provided by `y`
/// element wise.
///
/// ```py
/// D: int
/// x: [T; D]
/// y: [T; D]
///
/// for i in 0..D:
///     x[i] = x[i] + y[i]
/// ```
///
/// # Safety
///
/// Lengths of `x` and `y` **MUST** be equal.
pub unsafe fn generic_xany_fallback_nofma_sub_vertical<T>(x: &mut [T], y: &[T])
where
    T: Copy,
    AutoMath: Math<T>,
{
    f32_op_vertical(AutoMath::sub, x, y)
}

#[inline(always)]
unsafe fn f32_op_vertical<T, O>(op: O, x: &mut [T], y: &[T])
where
    T: Copy,
    O: Fn(T, T) -> T,
{
    debug_assert_eq!(x.len(), y.len());

    let len = x.len();
    let offset_from = x.len() % 8;

    let mut i = 0;
    while i < (len - offset_from) {
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

        *x.get_unchecked_mut(i) = op(x1, y1);
        *x.get_unchecked_mut(i + 1) = op(x2, y2);
        *x.get_unchecked_mut(i + 2) = op(x3, y3);
        *x.get_unchecked_mut(i + 3) = op(x4, y4);
        *x.get_unchecked_mut(i + 4) = op(x5, y5);
        *x.get_unchecked_mut(i + 5) = op(x6, y6);
        *x.get_unchecked_mut(i + 6) = op(x7, y7);
        *x.get_unchecked_mut(i + 7) = op(x8, y8);

        i += 8;
    }

    while i < len {
        let x = x.get_unchecked_mut(i);
        let y = *y.get_unchecked(i);

        *x = op(*x, y);

        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{assert_is_close_vector, get_sample_vectors};

    #[test]
    fn test_xany_nofma_div_vertical() {
        let (mut x, y) = get_sample_vectors(537);

        let expected_res = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| x / y)
            .collect::<Vec<f32>>();

        unsafe { generic_xany_fallback_nofma_div_vertical(&mut x, &y) };

        assert_is_close_vector(&x, &expected_res);
    }

    #[test]
    fn test_xany_nofma_mul_vertical() {
        let (mut x, y) = get_sample_vectors(537);

        let expected_res = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| x * y)
            .collect::<Vec<f32>>();

        unsafe { generic_xany_fallback_nofma_mul_vertical(&mut x, &y) };

        assert_is_close_vector(&x, &expected_res);
    }

    #[test]
    fn test_xany_nofma_add_vertical() {
        let (mut x, y) = get_sample_vectors(537);

        let expected_res = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| x + y)
            .collect::<Vec<f32>>();

        unsafe { generic_xany_fallback_nofma_add_vertical(&mut x, &y) };

        assert_is_close_vector(&x, &expected_res);
    }

    #[test]
    fn test_xany_nofma_sub_vertical() {
        let (mut x, y) = get_sample_vectors(537);

        let expected_res = x
            .iter()
            .zip(y.iter())
            .map(|(x, y)| x - y)
            .collect::<Vec<f32>>();

        unsafe { generic_xany_fallback_nofma_sub_vertical(&mut x, &y) };

        assert_is_close_vector(&x, &expected_res);
    }
}
