use std::ptr;

use cfavml::apply_dense;

/// Performs a copy on the given input buffer loading `N` rows of data
/// into the output buffer.
///
/// `offset` can be specified to shift what part of the data is loaded.
///
/// Internally `offset` is multiplied by `N` and `width`, so when `offset=1`
/// we select effectively the second sub buffer.
///
/// This assumes the buffer is row-major memory order.
pub unsafe fn select_row_major_row_wise_sub_buffer<T, const N: usize>(
    width: usize,
    offset: usize,
    input_buffer: &[T],
    output_buffer: &mut [T],
)
where
    T: Copy,
{
    debug_assert!(output_buffer.len() >= width * N, "Output buffer is not large enough");
    debug_assert!(input_buffer.len() >= width * N, "Input buffer is not large enough");
    debug_assert_ne!(N, 0, "N cannot be 0");
    debug_assert_ne!(width, 0, "Width cannot be 0");

    let true_offset = offset * N * width;

    let input_buffer = input_buffer.as_ptr();
    let output_buffer = output_buffer.as_mut_ptr();
    ptr::copy_nonoverlapping(input_buffer.add(true_offset), output_buffer, width * N);
}


/// Performs a copy on the given input buffer loading `N` columns of data
/// into the output buffer. The data itself is still row-wise but only
/// `N` elements per-row.
///
/// `offset` can be specified to shift what part of the data is loaded.
///
/// Internally `offset` is multiplied by `N`, so when `offset=1`
/// we select effectively the second sub buffer.
///
/// This assumes the buffer is row-major memory order.
pub unsafe fn select_row_major_column_row_blocks_sub_buffer<T, const N: usize>(
    height: usize,
    offset: usize,
    input_buffer: &[T],
    output_buffer: &mut [T],
)
where
    T: Copy,
{
    debug_assert!(output_buffer.len() >= height * N, "Output buffer is not large enough");
    debug_assert!(input_buffer.len() >= height * N, "Input buffer is not large enough");
    debug_assert_ne!(N, 0, "N cannot be 0");
    debug_assert_ne!(height, 0, "Height cannot be 0");
    debug_assert_eq!(input_buffer.len() % height, 0, "Input buffer has shape missmatch");

    let len = input_buffer.len();
    let width = input_buffer.len() / height;
    let true_offset = offset * N;

    let output_buffer = output_buffer.as_mut_ptr();
    let input_buffer = input_buffer.as_ptr();

    let mut step = 0;
    let mut x_offset = 0;
    while x_offset < len {
        ptr::copy_nonoverlapping(
            input_buffer.add(x_offset + true_offset),
            output_buffer.add(step),
            N,
        );

        x_offset += width;
        step += N;
    }
}


/// Transpose a row-major memory layout matrix to a column-major memory layout matrix.
pub unsafe fn transpose_matrix(
    width: usize,
    height: usize,
    input_buffer: &[f32],
    output_buffer: &mut [f32],
) {
    debug_assert_eq!(input_buffer.len(), output_buffer.len(), "Buffer sizes do not match");
    debug_assert_eq!(input_buffer.len(), width * height, "Matrix shape error");
    debug_assert_eq!(width % 8, 0, "Unsupported shape, TODO: Fix me"); // TODO: Fix me
    debug_assert_eq!(height % 8, 0, "Unsupported shape, TODO: Fix me"); // TODO: Fix me

    if is_x86_feature_detected!("avx2") {
        avx2_transpose::transpose_avx2_32bit(width, height, input_buffer, output_buffer)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2_transpose {
    use core::arch::x86_64::*;
    use std::mem;
    use cfavml::danger::DenseLane;

    #[allow(non_snake_case)]
    pub(crate) const fn _MM_SHUFFLE(z: u32, y: u32, x: u32, w: u32) -> i32 {
        ((z << 6) | (y << 4) | (x << 2) | w) as i32
    }

    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn transpose_avx2_32bit(
        width: usize,
        height: usize,
        data: &[f32],
        result: &mut [f32],
    ) {
        debug_assert_eq!(data.len(), width * height, "Input buffer has wrong shape");
        debug_assert_eq!(data.len(), result.len(), "Output buffers have size missmatch");

        let data_ptr = data.as_ptr();
        let result_ptr = result.as_mut_ptr();

        const BLOCK_STEP: usize = 16;

        // Step through our sub-blocks of the matrix.
        let mut i = 0;
        while i < width {
            let mut j = 0;
            while j < height {
                let dense1 = load_8x8_32bit(
                    i,
                    j * width,
                    width,
                    data_ptr,
                );
                let transposed1 = transpose_dense_32bit(dense1);

                let dense2 = load_8x8_32bit(
                    i + 8,
                    (j + 8) * width,
                    width,
                    data_ptr,
                );
                let transposed2 = transpose_dense_32bit(dense2);

                // Now our dense lanes have been transposed, our x_offset becomes our y_offset
                // and vice-versa.
                // TODO: Clean up
                _mm256_storeu_ps(result_ptr.add(j + (i * height)), transposed1.a);
                _mm256_storeu_ps(result_ptr.add(j + (i * height) + 8), transposed2.a);

                _mm256_storeu_ps(result_ptr.add(j + ((i + 1) * height)), transposed1.b);
                _mm256_storeu_ps(result_ptr.add(j + ((i + 1) * height) + 8), transposed2.b);

                _mm256_storeu_ps(result_ptr.add(j + ((i + 2) * height)), transposed1.c);
                _mm256_storeu_ps(result_ptr.add(j + ((i + 2) * height) + 8), transposed2.c);

                _mm256_storeu_ps(result_ptr.add(j + ((i + 3) * height)), transposed1.d);
                _mm256_storeu_ps(result_ptr.add(j + ((i + 3) * height) + 8), transposed2.d);

                _mm256_storeu_ps(result_ptr.add(j + ((i + 4) * height)), transposed1.e);
                _mm256_storeu_ps(result_ptr.add(j + ((i + 4) * height) + 8), transposed2.e);

                _mm256_storeu_ps(result_ptr.add(j + ((i + 5) * height)), transposed1.f);
                _mm256_storeu_ps(result_ptr.add(j + ((i + 5) * height) + 8), transposed2.f);

                _mm256_storeu_ps(result_ptr.add(j + ((i + 6) * height)), transposed1.g);
                _mm256_storeu_ps(result_ptr.add(j + ((i + 6) * height) + 8), transposed2.g);

                _mm256_storeu_ps(result_ptr.add(j + ((i + 7) * height)), transposed1.h);
                _mm256_storeu_ps(result_ptr.add(j + ((i + 7) * height) + 8), transposed2.h);

                j += BLOCK_STEP;
            }

            i += BLOCK_STEP;
        }
    }

    #[inline(always)]
    unsafe fn load_8x8_32bit(
        x_offset: usize,
        y_offset: usize,
        width: usize,
        data_ptr: *const f32,
    ) -> DenseLane<__m256> {
        // Load the 8x8 block of the matrix.
        DenseLane {
            a: _mm256_loadu_ps(data_ptr.add(y_offset + x_offset)),
            b: _mm256_loadu_ps(data_ptr.add(y_offset + x_offset + (width * 1))),
            c: _mm256_loadu_ps(data_ptr.add(y_offset + x_offset + (width * 2))),
            d: _mm256_loadu_ps(data_ptr.add(y_offset + x_offset + (width * 3))),
            e: _mm256_loadu_ps(data_ptr.add(y_offset + x_offset + (width * 4))),
            f: _mm256_loadu_ps(data_ptr.add(y_offset + x_offset + (width * 5))),
            g: _mm256_loadu_ps(data_ptr.add(y_offset + x_offset + (width * 6))),
            h: _mm256_loadu_ps(data_ptr.add(y_offset + x_offset + (width * 7))),
        }
    }

    #[inline(always)]
    unsafe fn transpose_dense_32bit(dense: DenseLane<__m256>) -> DenseLane<__m256> {
        let temp = DenseLane {
            // Unpack the low elements of the dense lane
            a: _mm256_unpacklo_ps(dense.a, dense.b),
            b: _mm256_unpacklo_ps(dense.c, dense.d),
            c: _mm256_unpacklo_ps(dense.e, dense.f),
            d: _mm256_unpacklo_ps(dense.g, dense.h),

            // Unpack the high elements of the dense lane
            e: _mm256_unpackhi_ps(dense.a, dense.b),
            f: _mm256_unpackhi_ps(dense.c, dense.d),
            g: _mm256_unpackhi_ps(dense.e, dense.f),
            h: _mm256_unpackhi_ps(dense.g, dense.h),
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

    #[cfg(all(
        test,
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    #[rustfmt::skip]
    mod tests {
        use std::mem;
        use super::*;

        #[test]
        fn test_32bit_8x8_transponse() {
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

            let expected: [f32; 64] = [
                1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
                2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7,
                4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7,
                5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7,
                6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7,
                7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7,
                8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7,
            ];

            unsafe {
                let dense = load_8x8_32bit(0, 0, 8, matrix.as_ptr());
                let transposed = transpose_dense_32bit(dense);
                let raw_data = mem::transmute::<_, [f32; 64]>(transposed);

                assert_eq!(&raw_data, &expected);
            }
        }

        #[test]
        fn test_32bit_transponse_8x8_small() {
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

            let mut result = [999.0; 64];

            let expected: [f32; 64] = [
                1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
                2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7,
                3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7,
                4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7,
                5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7,
                6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7,
                7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7,
                8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7,

            ];

            unsafe {
                transpose_avx2_32bit(8, 8, &matrix, &mut result);
                assert_eq!(&result, &expected);
            }
        }

        #[test]
        fn test_32bit_transponse_8x16_tall() {
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

            let expected: [f32; 128] = [
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
                transpose_avx2_32bit(8, 16, &matrix, &mut result);
                assert_eq!(&result, &expected);
            }
        }

        #[test]
        fn test_32bit_transponse_16x8_wide() {
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

            let expected: [f32; 128] = [
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
                transpose_avx2_32bit(16, 8, &matrix, &mut result);
                assert_eq!(&result, &expected);
            }
        }

        #[test]
        fn test_32bit_transponse_16x16_square() {
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

            let expected: [f32; 256] = [
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
                transpose_avx2_32bit(16, 16, &matrix, &mut result);
                assert_eq!(&result, &expected);
            }
        }
    }
}





#[cfg(test)]
#[rustfmt::skip]
mod tests {
    use super::*;

    #[test]
    fn test_select_row_major_row_wise_sub_buffer() {
        let test_buffer = &[
            0,  1,  2,  3,
            4,  5,  6,  7,
            8,  9,  10, 11,
            12, 13, 14, 15,
        ];

        let mut output = [999; 8];

        unsafe {
            select_row_major_row_wise_sub_buffer::<_, 2>(
                4,
                0,
                test_buffer,
                &mut output,
            );
        }
        let expected_buffer = &[
            0,  1,  2,  3,
            4,  5,  6,  7,
        ];
        assert_eq!(&output, expected_buffer);

        unsafe {
            select_row_major_row_wise_sub_buffer::<_, 2>(
                4,
                1,
                test_buffer,
                &mut output,
            );
        }
        let expected_buffer = &[
            8,  9,  10, 11,
            12, 13, 14, 15,
        ];
        assert_eq!(&output, expected_buffer);
    }

    #[test]
    fn test_select_row_major_column_row_blocks_sub_buffer() {
        let test_buffer = &[
            0,  1,  2,  3,
            4,  5,  6,  7,
            8,  9,  10, 11,
            12, 13, 14, 15,
        ];

        let mut output = [999; 8];

        unsafe {
            select_row_major_column_row_blocks_sub_buffer::<_, 2>(
                4,
                0,
                test_buffer,
                &mut output,
            );
        }
        let expected_buffer = &[
            0,  1,
            4,  5,
            8,  9,
            12, 13,
        ];
        assert_eq!(&output, expected_buffer);

        unsafe {
            select_row_major_column_row_blocks_sub_buffer::<_, 2>(
                4,
                1,
                test_buffer,
                &mut output,
            );
        }
        let expected_buffer = &[
            2,  3,
            6,  7,
            10, 11,
            14, 15,
        ];
        assert_eq!(&output, expected_buffer);
    }
}