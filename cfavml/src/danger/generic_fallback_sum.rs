use crate::math::*;

#[inline]
/// Sums all elements of the vector.
///
/// ```py
/// D: int
/// total: T
/// x: [T; D]
///
/// for i in 0..D:
///     total = total + x[i]
/// ```
///
/// # Safety
///
/// This method in theory is safe, but like the rest of the dangerous API, makes
/// no guarantee that it will always remain safe with no strings attached.
pub unsafe fn generic_xany_fallback_nofma_sum_horizontal<T>(x: &[T]) -> T
where
    T: Copy,
    AutoMath: Math<T>,
{
    sum::<T, AutoMath>(x)
}

#[allow(unused)]
#[inline]
/// Vertical sum of the given matrix returning the individual sums.
///
/// ```py
/// D: int
/// total: [T; D]
/// matrix: [[T; D]; N]
///
/// for i in 0..N:
///     for j in 0..D:
///         total[j] += matrix[i, j]   
/// ```
///
/// # Safety
///
/// All vectors within the matrix **MUST** be the same length.
pub unsafe fn generic_xany_fallback_nofma_sum_vertical<T>(matrix: &[&[T]]) -> Vec<T>
where
    T: Copy,
    AutoMath: Math<T>,
{
    sum_vertical::<T, AutoMath>(matrix)
}

#[inline(always)]
unsafe fn sum<T, M>(arr: &[T]) -> T
where
    T: Copy,
    M: Math<T>,
{
    let len = arr.len();
    let offset_from = len % 8;

    let mut extra = M::zero();
    let mut acc1 = M::zero();
    let mut acc2 = M::zero();
    let mut acc3 = M::zero();
    let mut acc4 = M::zero();
    let mut acc5 = M::zero();
    let mut acc6 = M::zero();
    let mut acc7 = M::zero();
    let mut acc8 = M::zero();

    let mut i = 0;
    while i < (len - offset_from) {
        let x1 = *arr.get_unchecked(i);
        let x2 = *arr.get_unchecked(i + 1);
        let x3 = *arr.get_unchecked(i + 2);
        let x4 = *arr.get_unchecked(i + 3);
        let x5 = *arr.get_unchecked(i + 4);
        let x6 = *arr.get_unchecked(i + 5);
        let x7 = *arr.get_unchecked(i + 6);
        let x8 = *arr.get_unchecked(i + 7);

        acc1 = M::add(acc1, x1);
        acc2 = M::add(acc2, x2);
        acc3 = M::add(acc3, x3);
        acc4 = M::add(acc4, x4);
        acc5 = M::add(acc5, x5);
        acc6 = M::add(acc6, x6);
        acc7 = M::add(acc7, x7);
        acc8 = M::add(acc8, x8);

        i += 8;
    }

    while i < len {
        let x = *arr.get_unchecked(i);
        extra = M::add(extra, x);

        i += 1;
    }

    acc1 = M::add(acc1, acc2);
    acc3 = M::add(acc3, acc4);
    acc5 = M::add(acc5, acc6);
    acc7 = M::add(acc7, acc8);

    acc1 = M::add(acc1, acc3);
    acc5 = M::add(acc5, acc7);

    M::add(M::add(acc1, acc5), extra)
}

#[inline(always)]
unsafe fn sum_vertical<T, M>(matrix: &[&[T]]) -> Vec<T>
where
    T: Copy,
    M: Math<T>,
{
    let len = matrix[0].len();
    let offset_from = len % 8;

    let mut results = vec![M::zero(); len];

    let mut i = 0;
    while i < offset_from {
        for m in 0..matrix.len() {
            let arr = *matrix.get_unchecked(m);
            debug_assert_eq!(arr.len(), len);

            let x = *arr.get_unchecked(i);
            let acc = *results.get_unchecked_mut(i);
            *results.get_unchecked_mut(i) = M::add(acc, x);
        }

        i += 1;
    }

    while i < len {
        let mut acc1 = M::zero();
        let mut acc2 = M::zero();
        let mut acc3 = M::zero();
        let mut acc4 = M::zero();
        let mut acc5 = M::zero();
        let mut acc6 = M::zero();
        let mut acc7 = M::zero();
        let mut acc8 = M::zero();

        for m in 0..matrix.len() {
            let arr = *matrix.get_unchecked(m);
            debug_assert_eq!(arr.len(), len);

            let x1 = *arr.get_unchecked(i);
            let x2 = *arr.get_unchecked(i + 1);
            let x3 = *arr.get_unchecked(i + 2);
            let x4 = *arr.get_unchecked(i + 3);
            let x5 = *arr.get_unchecked(i + 4);
            let x6 = *arr.get_unchecked(i + 5);
            let x7 = *arr.get_unchecked(i + 6);
            let x8 = *arr.get_unchecked(i + 7);

            acc1 = M::add(acc1, x1);
            acc2 = M::add(acc2, x2);
            acc3 = M::add(acc3, x3);
            acc4 = M::add(acc4, x4);
            acc5 = M::add(acc5, x5);
            acc6 = M::add(acc6, x6);
            acc7 = M::add(acc7, x7);
            acc8 = M::add(acc8, x8);
        }

        *results.get_unchecked_mut(i) = acc1;
        *results.get_unchecked_mut(i + 1) = acc2;
        *results.get_unchecked_mut(i + 2) = acc3;
        *results.get_unchecked_mut(i + 3) = acc4;
        *results.get_unchecked_mut(i + 4) = acc5;
        *results.get_unchecked_mut(i + 5) = acc6;
        *results.get_unchecked_mut(i + 6) = acc7;
        *results.get_unchecked_mut(i + 7) = acc8;

        i += 8;
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{assert_is_close, get_sample_vectors};

    #[test]
    fn test_xany_nofma_sum() {
        let (x, _) = get_sample_vectors(131);
        let sum = unsafe { generic_xany_fallback_nofma_sum_horizontal(&x) };
        assert_is_close(sum, x.iter().sum::<f32>());
    }

    #[test]
    fn test_xany_nofma_sum_vertical() {
        let mut matrix = Vec::new();
        for _ in 0..25 {
            let (x, _) = get_sample_vectors(537);
            matrix.push(x);
        }

        let matrix_view = matrix.iter().map(|v| v.as_ref()).collect::<Vec<&[f32]>>();

        let mut expected_vertical_sum = vec![0.0; 537];
        for i in 0..537 {
            let mut sum = 0.0;
            for arr in matrix.iter() {
                sum += arr[i];
            }
            expected_vertical_sum[i] = sum;
        }

        let sum = unsafe { generic_xany_fallback_nofma_sum_vertical(&matrix_view) };
        assert_eq!(sum, expected_vertical_sum);
    }
}
