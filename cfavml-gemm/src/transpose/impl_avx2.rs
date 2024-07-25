use std::arch::x86_64::*;
use cfavml::danger::DenseLane;

#[target_feature(enable = "avx2")]
/// Transposes a matrix of 32bit values.
///
/// The core data type used is a f32 but any 32bit value works really.
pub(super) unsafe fn transpose_32bit(
    width: usize,
    height: usize,
    data: &[f32],
    result: &mut [f32],
) {
    let data_ptr = data.as_ptr();
    let result_ptr = result.as_mut_ptr();

    let width_remainder = width % 16;
    let height_remainder = height % 16;

    let mut j = 0;
    while j < (height - height_remainder) {
        let mut i = 0;
        while i < (width - width_remainder) {
            // top-left
            let l1 = load_8x8_32bit(
                i + j * width,
                width,
                data_ptr,
            );
            let l1_transpose = transpose_dense_32bit(l1);
            store_8x8_32bit(j + i * height, height, l1_transpose, result_ptr);

            // bottom-left
            let l1 = load_8x8_32bit(
                i + (j + 8) * width,
                width,
                data_ptr,
            );
            let l1_transpose = transpose_dense_32bit(l1);
            store_8x8_32bit((j + 8) + i * height, height, l1_transpose, result_ptr);

            // top-right
            let l1 = load_8x8_32bit(
                (i + 8) + j * width,
                width,
                data_ptr,
            );
            let l1_transpose = transpose_dense_32bit(l1);
            store_8x8_32bit(j + (i + 8) * height, height, l1_transpose, result_ptr);

            // bottom-right
            let l1 = load_8x8_32bit(
                (i + 8) + (j + 8) * width,
                width,
                data_ptr,
            );
            let l1_transpose = transpose_dense_32bit(l1);
            store_8x8_32bit((j + 8) + (i + 8) * height, height, l1_transpose, result_ptr);

            i += 16;
        }

        j += 16;
    }

    // Handles the tail of each row that does not fit within 8 wide blocks.
    let mut j_remainder = 0;
    while j_remainder < (height - height_remainder) {
        let mut i = width - width_remainder;
        while i < width {
            *result.get_unchecked_mut(i * height + j_remainder) = *data.get_unchecked(j_remainder * width + i);

            i += 1;
        }

        j_remainder += 1;
    }

    // Handles the tail of each column that does not fit within 8 wide blocks.
    while j < height {
        let mut i = 0;
        while i < width {
            *result.get_unchecked_mut(i * height + j) = *data.get_unchecked(j * width + i);

            i += 1;
        }

        j += 1;
    }


}


#[inline(always)]
unsafe fn load_8x8_32bit(
    offset: usize,
    width: usize,
    data_ptr: *const f32,
) -> DenseLane<__m256> {
    // Load the 8x8 block of the matrix.
    DenseLane {
        a: _mm256_loadu_ps(data_ptr.add(offset)),
        b: _mm256_loadu_ps(data_ptr.add(offset + (width * 1))),
        c: _mm256_loadu_ps(data_ptr.add(offset + (width * 2))),
        d: _mm256_loadu_ps(data_ptr.add(offset + (width * 3))),
        e: _mm256_loadu_ps(data_ptr.add(offset + (width * 4))),
        f: _mm256_loadu_ps(data_ptr.add(offset + (width * 5))),
        g: _mm256_loadu_ps(data_ptr.add(offset + (width * 6))),
        h: _mm256_loadu_ps(data_ptr.add(offset + (width * 7))),
    }
}

#[inline(always)]
unsafe fn load_16x8_32bit(
    offset: usize,
    width: usize,
    data_ptr: *const f32,
) -> [DenseLane<__m256>; 2] {
    // Each lese lane loads 16x4 values.
    let l1 = DenseLane {
        a: _mm256_loadu_ps(data_ptr.add(offset)),
        b: _mm256_loadu_ps(data_ptr.add(offset + 8)),
        c: _mm256_loadu_ps(data_ptr.add(offset + (width * 1))),
        d: _mm256_loadu_ps(data_ptr.add(offset + (width * 1) + 8)),
        e: _mm256_loadu_ps(data_ptr.add(offset + (width * 2))),
        f: _mm256_loadu_ps(data_ptr.add(offset + (width * 2) + 8)),
        g: _mm256_loadu_ps(data_ptr.add(offset + (width * 3))),
        h: _mm256_loadu_ps(data_ptr.add(offset + (width * 3) + 8)),
    };

    let l2 = DenseLane {
        a: _mm256_loadu_ps(data_ptr.add(offset + (width * 4))),
        b: _mm256_loadu_ps(data_ptr.add(offset + (width * 4) + 8)),
        c: _mm256_loadu_ps(data_ptr.add(offset + (width * 5))),
        d: _mm256_loadu_ps(data_ptr.add(offset + (width * 5) + 8)),
        e: _mm256_loadu_ps(data_ptr.add(offset + (width * 6))),
        f: _mm256_loadu_ps(data_ptr.add(offset + (width * 6) + 8)),
        g: _mm256_loadu_ps(data_ptr.add(offset + (width * 7))),
        h: _mm256_loadu_ps(data_ptr.add(offset + (width * 7) + 8)),
    };

    // We then remap the values so we can transpose it as a 16x8 matrix.
    [
        DenseLane {
            a: l1.a,
            b: l1.c,
            c: l1.e,
            d: l1.g,
            e: l2.a,
            f: l2.c,
            g: l2.e,
            h: l2.g,
        },
        DenseLane {
            a: l1.b,
            b: l1.d,
            c: l1.f,
            d: l1.h,
            e: l2.b,
            f: l2.d,
            g: l2.f,
            h: l2.h,
        },
    ]
}

#[inline(always)]
unsafe fn store_8x8_32bit(
    offset: usize,
    height: usize,
    l1: DenseLane<__m256>,
    result_ptr: *mut f32,
) {
    _mm256_storeu_ps(result_ptr.add(offset), l1.a);
    _mm256_storeu_ps(result_ptr.add(offset + (1 * height)), l1.b);
    _mm256_storeu_ps(result_ptr.add(offset + (2 * height)), l1.c);
    _mm256_storeu_ps(result_ptr.add(offset + (3 * height)), l1.d);
    _mm256_storeu_ps(result_ptr.add(offset + (4 * height)), l1.e);
    _mm256_storeu_ps(result_ptr.add(offset + (5 * height)), l1.f);
    _mm256_storeu_ps(result_ptr.add(offset + (6 * height)), l1.g);
    _mm256_storeu_ps(result_ptr.add(offset + (7 * height)), l1.h);
}

#[inline(always)]
unsafe fn prefetch_8x8_32bit(
    offset: usize,
    result_ptr: *mut f32,
) {
    _mm_prefetch::<_MM_HINT_T1>(result_ptr.add(offset).cast());
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
            let dense = load_8x8_32bit(0, 8, matrix.as_ptr());
            let transposed = transpose_dense_32bit(dense);
            let raw_data = mem::transmute::<_, [f32; 64]>(transposed);

            assert_eq!(&raw_data, &expected);
        }
    }

    #[test]
    fn test_32bit_transponse_4x4_small() {
        let matrix = [
            1.0, 2.0, 3.0, 4.0,
            1.1, 2.1, 3.1, 4.1,
            1.2, 2.2, 3.2, 4.2,
            1.3, 2.3, 3.3, 4.3,
        ];

        let mut result = [999.0; 16];

        let expected: [f32; 16] = [
            1.0, 1.1, 1.2, 1.3,
            2.0, 2.1, 2.2, 2.3,
            3.0, 3.1, 3.2, 3.3,
            4.0, 4.1, 4.2, 4.3,
        ];

        unsafe {
            transpose_32bit(4, 4, &matrix, &mut result);
            assert_eq!(&result, &expected);
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

        let mut result = [999.0; 64];
        unsafe {
            transpose_32bit(8, 8, &matrix, &mut result);
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
            transpose_32bit(8, 16, &matrix, &mut result);
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
            transpose_32bit(16, 8, &matrix, &mut result);
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
            transpose_32bit(16, 16, &matrix, &mut result);
            assert_eq!(&result, &expected);
        }
    }

    #[test]
    fn test_32bit_transponse_40x40_square() {
        let (matrix, _) = crate::test_utils::get_sample_vectors(40 * 40);
        let mut result = vec![0.0; 40 * 40];
        let expected = crate::test_utils::basic_transpose(40, 40, &matrix);

        unsafe {
            transpose_32bit(40, 40, &matrix, &mut result);
            assert_eq!(&result, &expected);
        }
    }

    #[test]
    fn test_32bit_transponse_64x64_square() {
        let (matrix, _) = crate::test_utils::get_sample_vectors(64 * 64);
        let mut result = vec![0.0; 64 * 64];
        let expected = crate::test_utils::basic_transpose(64, 64, &matrix);

        unsafe {
            transpose_32bit(64, 64, &matrix, &mut result);
            assert_eq!(&result, &expected);
        }
    }

    #[test]
    fn test_32bit_transponse_4x40_long_no_block() {
        let (matrix, _) = crate::test_utils::get_sample_vectors(4 * 40);
        let mut result = vec![0.0; 4 * 40];
        let expected = crate::test_utils::basic_transpose(4, 40, &matrix);

        unsafe {
            transpose_32bit(4, 40, &matrix, &mut result);
            assert_eq!(&result, &expected);
        }
    }

    #[test]
    fn test_32bit_transponse_19x43_odd_sizes() {
        let (matrix, _) = crate::test_utils::get_sample_vectors(19 * 43);
        let mut result = vec![0.0; 19 * 43];
        let expected = crate::test_utils::basic_transpose(19, 43, &matrix);

        unsafe {
            transpose_32bit(19, 43, &matrix, &mut result);
            assert_eq!(&result, &expected);
        }
    }
}