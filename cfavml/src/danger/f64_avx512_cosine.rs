use crate::danger::utils::cosine;
use crate::danger::{
    f64_xany_avx512_fma_dot,
    f64_xany_avx512_fma_norm,
    f64_xconst_avx512_fma_dot,
    f64_xconst_avx512_fma_norm,
};
use crate::math::*;

#[target_feature(enable = "avx512f")]
#[inline]
/// Computes the cosine distance of two `[f64; DIMS]` vectors.
///
/// # Safety
///
/// DIMS **MUST** be a multiple of `64` and vectors must be `DIMS` in length,
/// otherwise this routine will become immediately UB due to out of bounds pointer accesses.
pub unsafe fn f64_xconst_avx512_fma_cosine<const DIMS: usize>(
    x: &[f64],
    y: &[f64],
) -> f64 {
    debug_assert_eq!(DIMS % 64, 0);
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), DIMS);

    let norm_x = f64_xconst_avx512_fma_norm::<DIMS>(x);
    let norm_y = f64_xconst_avx512_fma_norm::<DIMS>(y);
    let dot_product = f64_xconst_avx512_fma_dot::<DIMS>(x, y);

    cosine::<f64, AutoMath>(dot_product, norm_x, norm_y)
}

#[target_feature(enable = "avx512f")]
#[inline]
/// Computes the cosine distance of two `f64` vectors.
///
/// # Safety
///
/// Vectors **MUST** be equal length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
pub unsafe fn f64_xany_avx512_fma_cosine(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());

    let norm_x = f64_xany_avx512_fma_norm(x);
    let norm_y = f64_xany_avx512_fma_norm(y);
    let dot_product = f64_xany_avx512_fma_dot(x, y);

    cosine::<f64, AutoMath>(dot_product, norm_x, norm_y)
}

#[cfg(all(test, target_feature = "avx512f"))]
mod tests {
    use super::*;
    use crate::test_utils::{assert_is_close, get_sample_vectors, simple_cosine};

    #[test]
    fn test_xconst_fma_cosine() {
        let (x, y) = get_sample_vectors(1024);
        let dist = unsafe { f64_xconst_avx512_fma_cosine::<1024>(&x, &y) };
        assert_is_close(dist as f32, simple_cosine(&x, &y) as f32);
    }

    #[test]
    fn test_xany_fma_cosine() {
        let (x, y) = get_sample_vectors(547);
        let dist = unsafe { f64_xany_avx512_fma_cosine(&x, &y) };
        assert_is_close(dist as f32, simple_cosine(&x, &y) as f32);
    }
}
