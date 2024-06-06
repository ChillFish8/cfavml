use crate::danger::utils::cosine;
use crate::danger::{
    f32_xany_avx2_fma_dot,
    f32_xany_avx2_fma_norm,
    f32_xany_avx2_nofma_dot,
    f32_xany_avx2_nofma_norm,
    f32_xconst_avx2_fma_dot,
    f32_xconst_avx2_fma_norm,
    f32_xconst_avx2_nofma_dot,
    f32_xconst_avx2_nofma_norm,
};
use crate::math::*;

#[target_feature(enable = "avx2")]
#[inline]
/// Computes the cosine distance of two `[f32; DIMS]` vectors.
///
/// # Safety
///
/// DIMS **MUST** be a multiple of `64` and both vectors must be `DIMS` in length,
/// otherwise this routine will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_xconst_avx2_nofma_cosine<const DIMS: usize>(
    x: &[f32],
    y: &[f32],
) -> f32 {
    debug_assert_eq!(DIMS % 64, 0);
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), DIMS);

    let norm_x = f32_xconst_avx2_nofma_norm::<DIMS>(x);
    let norm_y = f32_xconst_avx2_nofma_norm::<DIMS>(y);
    let dot_product = f32_xconst_avx2_nofma_dot::<DIMS>(x, y);

    cosine::<f32, AutoMath>(dot_product, norm_x, norm_y)
}

#[target_feature(enable = "avx2")]
#[inline]
/// Computes the cosine distance of two `f32` vectors.
///
/// # Safety
///
/// Vectors **MUST** be the same length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_xany_avx2_nofma_cosine(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());

    let norm_x = f32_xany_avx2_nofma_norm(x);
    let norm_y = f32_xany_avx2_nofma_norm(y);
    let dot_product = f32_xany_avx2_nofma_dot(x, y);

    cosine::<f32, AutoMath>(dot_product, norm_x, norm_y)
}

#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
/// Computes the cosine distance of two `[f32; DIMS]` vectors.
///
/// # Safety
///
/// DIMS **MUST** be a multiple of `64` and both vectors must be `DIMS` in length,
/// otherwise this routine will become immediately UB due to out of bounds pointer accesses.
///
/// Values within the vector should also be finite and not `NaN`
pub unsafe fn f32_xconst_avx2_fma_cosine<const DIMS: usize>(
    x: &[f32],
    y: &[f32],
) -> f32 {
    debug_assert_eq!(DIMS % 64, 0);
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), DIMS);

    let norm_x = f32_xconst_avx2_fma_norm::<DIMS>(x);
    let norm_y = f32_xconst_avx2_fma_norm::<DIMS>(y);
    let dot_product = f32_xconst_avx2_fma_dot::<DIMS>(x, y);

    cosine::<f32, AutoMath>(dot_product, norm_x, norm_y)
}

#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
/// Computes the cosine distance of two `f32` vectors.
///
/// # Safety
///
/// Vectors **MUST** be the same length, otherwise this routine
/// will become immediately UB due to out of bounds pointer accesses.
///
/// NOTE:
/// Values within the vector should also be finite, although it is not
/// going to crash the program, it is going to produce insane numbers.
pub unsafe fn f32_xany_avx2_fma_cosine(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());

    let norm_x = f32_xany_avx2_fma_norm(x);
    let norm_y = f32_xany_avx2_fma_norm(y);
    let dot_product = f32_xany_avx2_fma_dot(x, y);

    cosine::<f32, AutoMath>(dot_product, norm_x, norm_y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{get_sample_vectors, is_close, simple_cosine};

    #[test]
    fn test_xany_fma_cosine() {
        let (x, y) = get_sample_vectors(127);
        let dist = unsafe { f32_xany_avx2_fma_cosine(&x, &y) };
        assert!(is_close(dist, simple_cosine(&x, &y)))
    }

    #[test]
    fn test_xany_nofma_cosine() {
        let (x, y) = get_sample_vectors(127);
        let dist = unsafe { f32_xany_avx2_nofma_cosine(&x, &y) };
        assert!(is_close(dist, simple_cosine(&x, &y)))
    }

    #[test]
    fn test_xconst_fma_cosine() {
        let (x, y) = get_sample_vectors(1024);
        let dist = unsafe { f32_xconst_avx2_fma_cosine::<1024>(&x, &y) };
        assert!(is_close(dist, simple_cosine(&x, &y)))
    }

    #[test]
    fn test_xconst_nofma_cosine() {
        let (x, y) = get_sample_vectors(1024);
        let dist = unsafe { f32_xconst_avx2_nofma_cosine::<1024>(&x, &y) };
        assert!(is_close(dist, simple_cosine(&x, &y)))
    }
}
