use crate::danger::{cosine, generic_xany_fallback_nofma_dot};
use crate::math::*;

#[inline]
/// Computes the cosine distance of two `T` vectors.
///
/// These are fallback routines, they are designed to be optimized
/// by the compiler only, in areas where manually optimized routines
/// are unable to run due to lack of CPU features.
///
/// # Safety
///
/// Vectors **MUST** be equal length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
pub unsafe fn generic_xany_fallback_nofma_cosine<T>(x: &[T], y: &[T]) -> T
where
    T: Copy,
    AutoMath: Math<T>,
{
    let norm_x = generic_xany_fallback_nofma_dot(x, x);
    let norm_y = generic_xany_fallback_nofma_dot(y, y);
    let dot_product = generic_xany_fallback_nofma_dot(x, y);
    cosine::<T, AutoMath>(dot_product, norm_x, norm_y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{assert_is_close, get_sample_vectors, simple_cosine};

    #[test]
    fn test_f32_xany_nofma_dot() {
        let (x, y) = get_sample_vectors(514);
        let dist = unsafe { generic_xany_fallback_nofma_cosine(&x, &y) };
        let expected = simple_cosine(&x, &y);
        assert_is_close(dist, expected);
    }
}
