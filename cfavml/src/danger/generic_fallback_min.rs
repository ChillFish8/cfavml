use crate::math::*;

#[inline]
/// Computes the horizontal minimum of the given vector that is `[T; N]`.
///
/// # Safety
///
/// This method in theory is safe, but like the rest of the dangerous API, makes
/// no guarantee that it will always remain safe with no strings attached.
pub unsafe fn f32_xany_fallback_nofma_min_horizontal<T>(arr: &[T]) -> T
where
    T: Copy,
    AutoMath: Math<T>,
{
    let len = arr.len();
    let offset_from = len % 8;

    let mut acc1 = AutoMath::max();
    let mut acc2 = AutoMath::max();
    let mut acc3 = AutoMath::max();
    let mut acc4 = AutoMath::max();
    let mut acc5 = AutoMath::max();
    let mut acc6 = AutoMath::max();
    let mut acc7 = AutoMath::max();
    let mut acc8 = AutoMath::max();

    let mut i = 0;
    while i < offset_from {
        let x = *arr.get_unchecked(i);
        acc1 = AutoMath::cmp_min(acc1, x);

        i += 1;
    }

    while i < len {
        let x1 = *arr.get_unchecked(i);
        let x2 = *arr.get_unchecked(i + 1);
        let x3 = *arr.get_unchecked(i + 2);
        let x4 = *arr.get_unchecked(i + 3);
        let x5 = *arr.get_unchecked(i + 4);
        let x6 = *arr.get_unchecked(i + 5);
        let x7 = *arr.get_unchecked(i + 6);
        let x8 = *arr.get_unchecked(i + 7);

        acc1 = AutoMath::cmp_min(acc1, x1);
        acc2 = AutoMath::cmp_min(acc2, x2);
        acc3 = AutoMath::cmp_min(acc3, x3);
        acc4 = AutoMath::cmp_min(acc4, x4);
        acc5 = AutoMath::cmp_min(acc5, x5);
        acc6 = AutoMath::cmp_min(acc6, x6);
        acc7 = AutoMath::cmp_min(acc7, x7);
        acc8 = AutoMath::cmp_min(acc8, x8);

        i += 8;
    }

    acc1 = AutoMath::cmp_min(acc1, acc2);
    acc3 = AutoMath::cmp_min(acc3, acc4);
    acc5 = AutoMath::cmp_min(acc5, acc6);
    acc7 = AutoMath::cmp_min(acc7, acc8);

    acc1 = AutoMath::cmp_min(acc1, acc3);
    acc5 = AutoMath::cmp_min(acc5, acc7);

    AutoMath::cmp_min(acc1, acc5)
}

#[allow(unused)]
#[inline]
/// Computes the horizontal minimum of the given vector that is `[[T; DIMS]; N]`.
///
/// # Safety
///
/// Each vector in the matrix must be the same size, this routine assumes the dimensions
/// of all vectors in the matrix are equal to the dimensions of the first vector in
/// the matrix.
pub unsafe fn generic_xany_fallback_nofma_min_vertical<T>(matrix: &[&[T]]) -> Vec<T>
where
    T: Copy,
    AutoMath: Math<T>,
{
    let len = matrix[0].len();

    let mut min_values = vec![AutoMath::zero(); len];
    let mut offset_from = len % 8;

    // We work our way horizontally by taking steps of 8 and finding
    // the min of for each of the lanes vertically through the matrix.
    let mut i = 0;
    while i < (len - offset_from) {
        let mut acc1 = AutoMath::max();
        let mut acc2 = AutoMath::max();
        let mut acc3 = AutoMath::max();
        let mut acc4 = AutoMath::max();
        let mut acc5 = AutoMath::max();
        let mut acc6 = AutoMath::max();
        let mut acc7 = AutoMath::max();
        let mut acc8 = AutoMath::max();

        // Vertical min of the 8 elements.
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

            acc1 = AutoMath::cmp_min(acc1, x1);
            acc2 = AutoMath::cmp_min(acc2, x2);
            acc3 = AutoMath::cmp_min(acc3, x3);
            acc4 = AutoMath::cmp_min(acc4, x4);
            acc5 = AutoMath::cmp_min(acc5, x5);
            acc6 = AutoMath::cmp_min(acc6, x6);
            acc7 = AutoMath::cmp_min(acc7, x7);
            acc8 = AutoMath::cmp_min(acc8, x8);
        }

        *min_values.get_unchecked_mut(i) = acc1;
        *min_values.get_unchecked_mut(i + 1) = acc2;
        *min_values.get_unchecked_mut(i + 2) = acc3;
        *min_values.get_unchecked_mut(i + 3) = acc4;
        *min_values.get_unchecked_mut(i + 4) = acc5;
        *min_values.get_unchecked_mut(i + 5) = acc6;
        *min_values.get_unchecked_mut(i + 6) = acc7;
        *min_values.get_unchecked_mut(i + 7) = acc8;

        i += 8;
    }

    while i < len {
        let mut acc = AutoMath::max();
        for m in 0..matrix.len() {
            let arr = *matrix.get_unchecked(m);
            debug_assert_eq!(arr.len(), len);

            let x = *arr.get_unchecked(i);
            acc = AutoMath::cmp_min(acc, x);
        }

        *min_values.get_unchecked_mut(i) = acc;

        i += 1;
    }

    min_values
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::get_sample_vectors;

    #[test]
    fn test_xany_nofma_min_horizontal() {
        let (x, _) = get_sample_vectors(131);
        let min = unsafe { f32_xany_fallback_nofma_min_horizontal(&x) };
        assert_eq!(min, x.iter().fold(f32::INFINITY, |acc, v| acc.min(*v)));
    }

    #[test]
    fn test_xany_nofma_min_vertical() {
        let mut matrix = Vec::new();
        for _ in 0..25 {
            let (x, _) = get_sample_vectors(537);
            matrix.push(x);
        }

        let matrix_view = matrix.iter().map(|v| v.as_ref()).collect::<Vec<&[f32]>>();

        let mut expected_vertical_min = vec![f32::INFINITY; 537];
        for i in 0..537 {
            let mut min = f32::INFINITY;
            for arr in matrix.iter() {
                min = min.min(arr[i]);
            }
            expected_vertical_min[i] = min;
        }

        let min = unsafe { generic_xany_fallback_nofma_min_vertical(&matrix_view) };
        assert_eq!(min, expected_vertical_min);
    }
}
