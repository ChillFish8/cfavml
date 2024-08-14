use std::arch::x86_64::*;

use cfavml::danger::*;

use crate::transpose::{generic_transpose, TransposeMatrix};

#[inline]
#[target_feature(enable = "avx2")]
/// Performs a matrix transposition on 32 bit values.
///
/// # Safety
///
/// The size of the input and output buffers _must_ be equal to the calculated size by doing
/// `width * height`.
///
/// This function also assumes `avx2` CPU features are available.
pub unsafe fn f32_xany_avx2_transpose(
    width: usize,
    height: usize,
    data: &[f32],
    result: &mut [f32],
) {
    generic_transpose::<f32, Avx2>(width, height, data, result)
}

#[inline]
#[target_feature(enable = "avx2")]
/// Performs a matrix transposition on 64 bit values.
///
/// # Safety
///
/// The size of the input and output buffers _must_ be equal to the calculated size by doing
/// `width * height`.
///
/// This function also assumes `avx2` CPU features are available.
pub unsafe fn f64_xany_avx2_transpose(
    width: usize,
    height: usize,
    data: &[f64],
    result: &mut [f64],
) {
    generic_transpose::<f64, Avx2>(width, height, data, result)
}

impl TransposeMatrix<f32> for Avx2 {
    type RegisterMatrix = DenseLane<Self::Register>;

    #[inline(always)]
    unsafe fn load_matrix(
        offset: usize,
        width: usize,
        data_ptr: *const f32,
    ) -> Self::RegisterMatrix {
        DenseLane {
            a: Self::load(data_ptr.add(offset)),
            b: Self::load(data_ptr.add(offset + (width * 1))),
            c: Self::load(data_ptr.add(offset + (width * 2))),
            d: Self::load(data_ptr.add(offset + (width * 3))),
            e: Self::load(data_ptr.add(offset + (width * 4))),
            f: Self::load(data_ptr.add(offset + (width * 5))),
            g: Self::load(data_ptr.add(offset + (width * 6))),
            h: Self::load(data_ptr.add(offset + (width * 7))),
        }
    }

    #[inline(always)]
    unsafe fn write_matrix(
        offset: usize,
        height: usize,
        matrix: Self::RegisterMatrix,
        result_ptr: *mut f32,
    ) {
        Self::write(result_ptr.add(offset), matrix.a);
        Self::write(result_ptr.add(offset + (1 * height)), matrix.b);
        Self::write(result_ptr.add(offset + (2 * height)), matrix.c);
        Self::write(result_ptr.add(offset + (3 * height)), matrix.d);
        Self::write(result_ptr.add(offset + (4 * height)), matrix.e);
        Self::write(result_ptr.add(offset + (5 * height)), matrix.f);
        Self::write(result_ptr.add(offset + (6 * height)), matrix.g);
        Self::write(result_ptr.add(offset + (7 * height)), matrix.h);
    }

    #[inline(always)]
    unsafe fn transpose_register_matrix(
        matrix: Self::RegisterMatrix,
    ) -> Self::RegisterMatrix {
        let temp = DenseLane {
            // Unpack the low elements of the dense lane
            a: _mm256_unpacklo_ps(matrix.a, matrix.b),
            b: _mm256_unpacklo_ps(matrix.c, matrix.d),
            c: _mm256_unpacklo_ps(matrix.e, matrix.f),
            d: _mm256_unpacklo_ps(matrix.g, matrix.h),

            // Unpack the high elements of the dense lane
            e: _mm256_unpackhi_ps(matrix.a, matrix.b),
            f: _mm256_unpackhi_ps(matrix.c, matrix.d),
            g: _mm256_unpackhi_ps(matrix.e, matrix.f),
            h: _mm256_unpackhi_ps(matrix.g, matrix.h),
        };

        let half_transposed = DenseLane {
            a: _mm256_shuffle_ps::<0x44>(temp.a, temp.b),
            b: _mm256_shuffle_ps::<0xEE>(temp.a, temp.b),
            c: _mm256_shuffle_ps::<0x44>(temp.c, temp.d),
            d: _mm256_shuffle_ps::<0xEE>(temp.c, temp.d),
            e: _mm256_shuffle_ps::<0x44>(temp.e, temp.f),
            f: _mm256_shuffle_ps::<0xEE>(temp.e, temp.f),
            g: _mm256_shuffle_ps::<0x44>(temp.g, temp.h),
            h: _mm256_shuffle_ps::<0xEE>(temp.g, temp.h),
        };

        DenseLane {
            a: _mm256_permute2f128_ps::<0x20>(half_transposed.a, half_transposed.c),
            b: _mm256_permute2f128_ps::<0x20>(half_transposed.b, half_transposed.d),
            c: _mm256_permute2f128_ps::<0x20>(half_transposed.e, half_transposed.g),
            d: _mm256_permute2f128_ps::<0x20>(half_transposed.f, half_transposed.h),
            e: _mm256_permute2f128_ps::<0x31>(half_transposed.a, half_transposed.c),
            f: _mm256_permute2f128_ps::<0x31>(half_transposed.b, half_transposed.d),
            g: _mm256_permute2f128_ps::<0x31>(half_transposed.e, half_transposed.g),
            h: _mm256_permute2f128_ps::<0x31>(half_transposed.f, half_transposed.h),
        }
    }
}

#[derive(Copy, Clone)]
pub(crate) struct Dense4x4Lane<T> {
    pub a: T,
    pub b: T,
    pub c: T,
    pub d: T,
}

impl TransposeMatrix<f64> for Avx2 {
    type RegisterMatrix = Dense4x4Lane<Self::Register>;

    #[inline(always)]
    unsafe fn load_matrix(
        offset: usize,
        width: usize,
        data_ptr: *const f64,
    ) -> Self::RegisterMatrix {
        Dense4x4Lane {
            a: Self::load(data_ptr.add(offset)),
            b: Self::load(data_ptr.add(offset + (width * 1))),
            c: Self::load(data_ptr.add(offset + (width * 2))),
            d: Self::load(data_ptr.add(offset + (width * 3))),
        }
    }

    #[inline(always)]
    unsafe fn write_matrix(
        offset: usize,
        height: usize,
        matrix: Self::RegisterMatrix,
        result_ptr: *mut f64,
    ) {
        Self::write(result_ptr.add(offset), matrix.a);
        Self::write(result_ptr.add(offset + (1 * height)), matrix.b);
        Self::write(result_ptr.add(offset + (2 * height)), matrix.c);
        Self::write(result_ptr.add(offset + (3 * height)), matrix.d);
    }

    #[inline(always)]
    unsafe fn transpose_register_matrix(
        matrix: Self::RegisterMatrix,
    ) -> Self::RegisterMatrix {
        let temp = Dense4x4Lane {
            // Unpack the low elements of the dense lane
            a: _mm256_unpacklo_pd(matrix.a, matrix.b),
            b: _mm256_unpacklo_pd(matrix.c, matrix.d),

            // Unpack the high elements of the dense lane
            c: _mm256_unpackhi_pd(matrix.a, matrix.b),
            d: _mm256_unpackhi_pd(matrix.c, matrix.d),
        };

        Dense4x4Lane {
            a: _mm256_permute2f128_pd::<{ _MM_SHUFFLE(0, 0, 0, 2) }>(temp.b, temp.a),
            b: _mm256_permute2f128_pd::<{ _MM_SHUFFLE(0, 0, 0, 2) }>(temp.d, temp.c),
            c: _mm256_permute2f128_pd::<{ _MM_SHUFFLE(0, 3, 0, 1) }>(temp.a, temp.b),
            d: _mm256_permute2f128_pd::<{ _MM_SHUFFLE(0, 3, 0, 1) }>(temp.c, temp.d),
        }
    }
}

#[allow(non_snake_case)]
const fn _MM_SHUFFLE(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[cfg(all(test, not(miri)))] // This is just very expensive to do
mod tests {
    use super::*;
    use crate::transpose::test_suite::run_test_suites_f32;

    #[test]
    fn test_avx2_f32() {
        run_test_suites_f32::<Avx2>();
    }
}
