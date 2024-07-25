use std::arch::x86_64::*;

const BLOCK_SIZE: usize = 4;

#[target_feature(enable = "avx2")]
/// Transposes a matrix of 64 bit values.
///
/// The core data type used is a f32 but any 64 bit value works really.
pub(super) unsafe fn transpose_64bit(
    width: usize,
    height: usize,
    data: &[f64],
    result: &mut [f64],
) {
    let data_ptr = data.as_ptr();
    let result_ptr = result.as_mut_ptr();

    let width_remainder = width % 8;
    let height_remainder = height % 8;

    let mut j = 0;
    while j < (height - height_remainder) {
        let mut i = 0;
        while i < (width - width_remainder) {
            // By performing a 8x8 matrix transposition made up of
            // 4x4 sub-operations we maintain cache locality as best we can
            // Increasing the size to 16x16 did not appear to provide and additional
            // benefit and was overall a net less according to benchmarks.

            // top-left
            let l1 = load_4x4_64bit(i + j * width, width, data_ptr);
            let l1_transpose = transpose_dense_64bit(l1);
            store_4x4_64bit(j + i * height, height, l1_transpose, result_ptr);

            // bottom-left
            let l1 = load_4x4_64bit(i + (j + BLOCK_SIZE) * width, width, data_ptr);
            let l1_transpose = transpose_dense_64bit(l1);
            store_4x4_64bit(
                (j + BLOCK_SIZE) + i * height,
                height,
                l1_transpose,
                result_ptr,
            );

            // top-right
            let l1 = load_4x4_64bit((i + BLOCK_SIZE) + j * width, width, data_ptr);
            let l1_transpose = transpose_dense_64bit(l1);
            store_4x4_64bit(
                j + (i + BLOCK_SIZE) * height,
                height,
                l1_transpose,
                result_ptr,
            );

            // bottom-right
            let l1 = load_4x4_64bit(
                (i + BLOCK_SIZE) + (j + BLOCK_SIZE) * width,
                width,
                data_ptr,
            );
            let l1_transpose = transpose_dense_64bit(l1);
            store_4x4_64bit(
                (j + BLOCK_SIZE) + (i + BLOCK_SIZE) * height,
                height,
                l1_transpose,
                result_ptr,
            );

            i += 8;
        }

        j += 8;
    }

    // Handles the tail of each row that does not fit within 8 wide blocks.
    let mut j_remainder = 0;
    while j_remainder < (height - height_remainder) {
        let mut i = width - width_remainder;
        while i < width {
            *result.get_unchecked_mut(i * height + j_remainder) =
                *data.get_unchecked(j_remainder * width + i);

            i += 1;
        }

        j_remainder += 1;
    }

    // Handles the tail of each column that does not fit within 8 wide blocks.
    while j < height {
        let mut i = 0;
        while i < width {
            *result.get_unchecked_mut(i * height + j) =
                *data.get_unchecked(j * width + i);

            i += 1;
        }

        j += 1;
    }
}

#[derive(Copy, Clone)]
struct Dense4x4Lane<T> {
    pub a: T,
    pub b: T,
    pub c: T,
    pub d: T,
}

#[inline(always)]
unsafe fn load_4x4_64bit(
    offset: usize,
    width: usize,
    data_ptr: *const f64,
) -> Dense4x4Lane<__m256d> {
    // Load the 4x4 block of the matrix.
    Dense4x4Lane {
        a: _mm256_loadu_pd(data_ptr.add(offset)),
        b: _mm256_loadu_pd(data_ptr.add(offset + (width * 1))),
        c: _mm256_loadu_pd(data_ptr.add(offset + (width * 2))),
        d: _mm256_loadu_pd(data_ptr.add(offset + (width * 3))),
    }
}

#[inline(always)]
unsafe fn store_4x4_64bit(
    offset: usize,
    height: usize,
    l1: Dense4x4Lane<__m256d>,
    result_ptr: *mut f64,
) {
    _mm256_storeu_pd(result_ptr.add(offset), l1.a);
    _mm256_storeu_pd(result_ptr.add(offset + (1 * height)), l1.b);
    _mm256_storeu_pd(result_ptr.add(offset + (2 * height)), l1.c);
    _mm256_storeu_pd(result_ptr.add(offset + (3 * height)), l1.d);
}

#[inline(always)]
unsafe fn transpose_dense_64bit(dense: Dense4x4Lane<__m256d>) -> Dense4x4Lane<__m256d> {
    let temp = Dense4x4Lane {
        // Unpack the low elements of the dense lane
        a: _mm256_unpacklo_pd(dense.a, dense.b),
        b: _mm256_unpacklo_pd(dense.c, dense.d),

        // Unpack the high elements of the dense lane
        c: _mm256_unpackhi_pd(dense.a, dense.b),
        d: _mm256_unpackhi_pd(dense.c, dense.d),
    };

    Dense4x4Lane {
        a: _mm256_permute2f128_pd::<{ _MM_SHUFFLE(0, 0, 0, 2) }>(temp.b, temp.a),
        b: _mm256_permute2f128_pd::<{ _MM_SHUFFLE(0, 0, 0, 2) }>(temp.d, temp.c),
        c: _mm256_permute2f128_pd::<{ _MM_SHUFFLE(0, 3, 0, 1) }>(temp.a, temp.b),
        d: _mm256_permute2f128_pd::<{ _MM_SHUFFLE(0, 3, 0, 1) }>(temp.c, temp.d),
    }
}

#[allow(non_snake_case)]
pub(crate) const fn _MM_SHUFFLE(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[rustfmt::skip]
mod tests {
    use std::mem;
    use super::*;

    #[test]
    fn test_64bit_8x8_transponse() {
        let matrix = [
            1.0, 2.0, 3.0, 4.0,
            1.1, 2.1, 3.1, 4.1,
            1.2, 2.2, 3.2, 4.2,
            1.3, 2.3, 3.3, 4.3,
        ];

        let expected = [
            1.0, 1.1, 1.2, 1.3,
            2.0, 2.1, 2.2, 2.3,
            3.0, 3.1, 3.2, 3.3,
            4.0, 4.1, 4.2, 4.3,
        ];

        unsafe {
            let dense = load_4x4_64bit(0, 4, matrix.as_ptr());
            let transposed = transpose_dense_64bit(dense);
            let raw_data = mem::transmute::<_, [f64; 16]>(transposed);

            assert_eq!(&raw_data, &expected);
        }
    }

    #[test]
    fn test_64bit_transponse_2x2_small() {
        let matrix = [
            1.0, 2.0,
            1.1, 2.1,
        ];

        let mut result = [999.0; 4];

        let expected = [
            1.0, 1.1,
            2.0, 2.1,
        ];

        unsafe {
            transpose_64bit(2, 2, &matrix, &mut result);
            assert_eq!(&result, &expected);
        }
    }

    #[test]
    fn test_64bit_transponse_8x8_small() {
        let matrix = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1,
            1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2,
            1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3,
            1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4,
            1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5,
            1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6,
            1.7, 2.7, 3.7, 4.7, 5.7, 6.7, 7.7, 8.7,
        ];

        let expected = [
            1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
            2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
            3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7,
            4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7,
            5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7,
            6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7,
            7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7,
            8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7,
        ];

        let mut result = [999.0; 64];
        unsafe {
            transpose_64bit(8, 8, &matrix, &mut result);
            assert_eq!(&result, &expected);
        }
    }

    #[test]
    fn test_64bit_transponse_8x16_tall() {
        let matrix = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1,
            1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2,
            1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3,
            1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4,
            1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5,
            1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6,
            1.7, 2.7, 3.7, 4.7, 5.7, 6.7, 7.7, 8.7,
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1,
            1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2,
            1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3,
            1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4,
            1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5,
            1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6,
            1.7, 2.7, 3.7, 4.7, 5.7, 6.7, 7.7, 8.7,
        ];

        let mut result = [999.0; 128];

        let expected = [
            1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
            2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
            3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7,
            4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7,
            5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7,
            6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7,
            7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7,
            8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7,
        ];

        unsafe {
            transpose_64bit(8, 16, &matrix, &mut result);
            assert_eq!(&result, &expected);
        }
    }

    #[test]
    fn test_64bit_transponse_16x8_wide() {
        let matrix = [
            1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
            2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
            3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7,
            4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7,
            5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7,
            6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7,
            7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7,
            8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7,
        ];

        let mut result = [999.0; 128];

        let expected = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1,
            1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2,
            1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3,
            1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4,
            1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5,
            1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6,
            1.7, 2.7, 3.7, 4.7, 5.7, 6.7, 7.7, 8.7,
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1,
            1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2,
            1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3,
            1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4,
            1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5,
            1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6,
            1.7, 2.7, 3.7, 4.7, 5.7, 6.7, 7.7, 8.7,
        ];

        unsafe {
            transpose_64bit(16, 8, &matrix, &mut result);
            assert_eq!(&result, &expected);
        }
    }

    #[test]
    fn test_64bit_transponse_16x16_square() {
        let matrix = [
            1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
            2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
            3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7,
            4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7,
            5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7,
            6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7,
            7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7,
            8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7,
            1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
            2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
            3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7,
            4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7,
            5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7,
            6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7,
            7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7,
            8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7,
        ];

        let mut result = [999.0; 256];

        let expected = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1,
            1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2,
            1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3,
            1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4,
            1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5,
            1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6, 1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6,
            1.7, 2.7, 3.7, 4.7, 5.7, 6.7, 7.7, 8.7, 1.7, 2.7, 3.7, 4.7, 5.7, 6.7, 7.7, 8.7,
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1,
            1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2,
            1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3, 8.3,
            1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4, 8.4,
            1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5,
            1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6, 1.6, 2.6, 3.6, 4.6, 5.6, 6.6, 7.6, 8.6,
            1.7, 2.7, 3.7, 4.7, 5.7, 6.7, 7.7, 8.7, 1.7, 2.7, 3.7, 4.7, 5.7, 6.7, 7.7, 8.7,
        ];

        unsafe {
            transpose_64bit(16, 16, &matrix, &mut result);
            assert_eq!(&result, &expected);
        }
    }

    #[test]
    fn test_64bit_transponse_40x40_square() {
        let (matrix, _) = crate::test_utils::get_sample_vectors(40 * 40);
        let mut result = vec![0.0; 40 * 40];
        let expected = crate::test_utils::basic_transpose(40, 40, &matrix);

        unsafe {
            transpose_64bit(40, 40, &matrix, &mut result);
            assert_eq!(&result, &expected);
        }
    }

    #[test]
    fn test_64bit_transponse_64x64_square() {
        let (matrix, _) = crate::test_utils::get_sample_vectors(64 * 64);
        let mut result = vec![0.0; 64 * 64];
        let expected = crate::test_utils::basic_transpose(64, 64, &matrix);

        unsafe {
            transpose_64bit(64, 64, &matrix, &mut result);
            assert_eq!(&result, &expected);
        }
    }

    #[test]
    fn test_64bit_transponse_4x40_long_no_block() {
        let (matrix, _) = crate::test_utils::get_sample_vectors(4 * 40);
        let mut result = vec![0.0; 4 * 40];
        let expected = crate::test_utils::basic_transpose(4, 40, &matrix);

        unsafe {
            transpose_64bit(4, 40, &matrix, &mut result);
            assert_eq!(&result, &expected);
        }
    }

    #[test]
    fn test_64bit_transponse_2x40_long_no_block() {
        let (matrix, _) = crate::test_utils::get_sample_vectors(2 * 40);
        let mut result = vec![0.0; 2 * 40];
        let expected = crate::test_utils::basic_transpose(2, 40, &matrix);

        unsafe {
            transpose_64bit(2, 40, &matrix, &mut result);
            assert_eq!(&result, &expected);
        }
    }

    #[test]
    fn test_64bit_transponse_19x43_odd_sizes() {
        let (matrix, _) = crate::test_utils::get_sample_vectors(19 * 43);
        let mut result = vec![0.0; 19 * 43];
        let expected = crate::test_utils::basic_transpose(19, 43, &matrix);

        unsafe {
            transpose_64bit(19, 43, &matrix, &mut result);
            assert_eq!(&result, &expected);
        }
    }
}
