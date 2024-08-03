use std::arch::x86_64::*;

use cfavml::danger::DenseLane;
use cfavml_utils::aligned_buffer::AlignedBuffer;

#[target_feature(enable = "avx2", enable = "fma")]
/// AVX2 + FMA enabled f32 matmul operation.
///
/// This routine internally implements a micro-kernel of `4x16` sub-matrices.
///
/// Internally this is formed of 8 x 256 bit registers storing the intermediate
/// results, the reason why we are doing `4x16` instead of `8x8` is because
/// we can do two FMADD operations for every single value broadcast of A we have to do.
/// Which in turn removes an additional mem load per step.
///
/// NOTE:
///
/// This routine assumes matrix `a` & `b` are **row-major** in layout.
///
/// # Safety
///
/// The `avx2` and `fma` CPU features must be supported by the CPU executing the
/// function, and the size of `a`, `b` and `result` must match the calculated shape
/// of their respective width and height parameters.
pub unsafe fn matmul(
    a_width: usize,
    a_height: usize,
    a: &[f32],
    b_width: usize,
    b_height: usize,
    b: &[f32],
    result: &mut [f32],
) {
    debug_assert_eq!(a.len(), a_width * a_height, "Input matrix `a` does not match provided shape");
    debug_assert_eq!(b.len(), b_width * b_height, "Input matrix `b` does not match provided shape");
    debug_assert_eq!(result.len(), a_width * b_height, "Input matrix `result` does not match expected result shape");

    assert_eq!(a_width % 16, 0, "Matrix shape must be a multiple of 16 atm"); // TODO: Remove
    assert_eq!(b_width % 16, 0, "Matrix shape must be a multiple of 16 atm"); // TODO: Remove
    assert_eq!(a_height % 16, 0, "Matrix shape must be a multiple of 16 atm"); // TODO: Remove
    assert_eq!(b_height % 16, 0, "Matrix shape must be a multiple of 16 atm"); // TODO: Remove

    const BLOCK_X: usize = 16;
    const BLOCK_Y: usize = 16;

    // - Optimize the B matrix memory layout to maximise the cache rate.
    // - Transpose matrix A so it is now column-major.
    let b_buffer = prep_column_matrix_component(b_width, b_height, b);

    let mut temp_buffer = unsafe { AlignedBuffer::<f32>::zeroed(BLOCK_Y * a_width) };
    let mut result_buffer = temp_buffer.clone();

    let b = b_buffer.as_slice();

    let b_ptr = b.as_ptr();
    let result_ptr = result.as_mut_ptr();

    // TODO: Implement this as a 16x16 kernel, operating in mini blocks of 4x16 ops.
    //       much cheaper to transpose and copy data to and from.
    let mut i = 0;
    while i < a_height {
        // Copy 4 x N rows of A and transpose.
        let a_read_offset = i * a_width;
        crate::transpose::transpose_matrix(
            a_width,
            16,
            a.get_unchecked(a_read_offset..a_read_offset+16*a_width),
            temp_buffer.as_mut_slice(),
        );

        let mut j = 0;
        while j < b_width {
            compute_16x16_matrix(
                temp_buffer.as_ptr(),
                b_ptr.add(j * b_height),
                result_ptr.add(j + i*a_width),
                a_width,
            );

            j += BLOCK_X;
        }

        i += BLOCK_Y;
    }
}


#[inline(always)]
unsafe fn compute_16x16_matrix(
    a_ptr: *const f32,
    b_ptr: *const f32,
    result_ptr: *mut f32,
    a_width: usize,
) {
    // A-B=row1 C-D=row2 E-F=row3 G-H=row4

    for block in 0..4 {
        let a_block_step = block * 4;
        let result_offset = block * 4 * a_width;

        let mut matrix = DenseLane::copy(_mm256_setzero_ps());

        for col in 0..a_width {
            let a_offset = (col * 16) + a_block_step;
            let b_offset = col * 16;

            let b_row1_part1 = _mm256_loadu_ps(b_ptr.add(b_offset + 0));
            let b_row1_part2 = _mm256_loadu_ps(b_ptr.add(b_offset + 8));

            // A Row 1 & 2
            let a_broadcast_col1_row1 = _mm256_set1_ps(a_ptr.add(a_offset + 0).read());
            let a_broadcast_col1_row2 = _mm256_set1_ps(a_ptr.add(a_offset + 1).read());
            matrix.a = _mm256_fmadd_ps(b_row1_part1, a_broadcast_col1_row1, matrix.a);
            matrix.b = _mm256_fmadd_ps(b_row1_part2, a_broadcast_col1_row1, matrix.b);
            matrix.c = _mm256_fmadd_ps(b_row1_part1, a_broadcast_col1_row2, matrix.c);
            matrix.d = _mm256_fmadd_ps(b_row1_part2, a_broadcast_col1_row2, matrix.d);

            // A Row 3 & 4
            let a_broadcast_col1_row3 = _mm256_set1_ps(a_ptr.add(a_offset + 2).read());
            let a_broadcast_col1_row4 = _mm256_set1_ps(a_ptr.add(a_offset + 3).read());
            matrix.e = _mm256_fmadd_ps(b_row1_part1, a_broadcast_col1_row3, matrix.e);
            matrix.f = _mm256_fmadd_ps(b_row1_part2, a_broadcast_col1_row3, matrix.f);
            matrix.g = _mm256_fmadd_ps(b_row1_part1, a_broadcast_col1_row4, matrix.g);
            matrix.h = _mm256_fmadd_ps(b_row1_part2, a_broadcast_col1_row4, matrix.h);
        }

        _mm256_storeu_ps(result_ptr.add(result_offset + (0 * a_width) + 0), matrix.a);
        _mm256_storeu_ps(result_ptr.add(result_offset + (0 * a_width) + 8), matrix.b);
        _mm256_storeu_ps(result_ptr.add(result_offset + (1 * a_width) + 0), matrix.c);
        _mm256_storeu_ps(result_ptr.add(result_offset + (1 * a_width) + 8), matrix.d);
        _mm256_storeu_ps(result_ptr.add(result_offset + (2 * a_width) + 0), matrix.e);
        _mm256_storeu_ps(result_ptr.add(result_offset + (2 * a_width) + 8), matrix.f);
        _mm256_storeu_ps(result_ptr.add(result_offset + (3 * a_width) + 0), matrix.g);
        _mm256_storeu_ps(result_ptr.add(result_offset + (3 * a_width) + 8), matrix.h);
    }
}

unsafe fn prep_column_matrix_component(width: usize, height: usize, data: &[f32]) -> AlignedBuffer<f32> {
    let mut layout_buffer = unsafe { AlignedBuffer::<f32>::zeroed(width * height) };
    copy_16wide_layout(width, height, data.as_ptr(), layout_buffer.as_mut_slice().as_mut_ptr());
    layout_buffer
}

unsafe fn prep_row_matrix_component(width: usize, height: usize, data: &[f32]) -> Vec<f32> {
    let mut layout_buffer = Vec::with_capacity(width * height);
    layout_buffer.copy_from_slice(data);  // TODO: Not sure how expensive this is.
    crate::transpose::transpose_matrix(width, height, data, layout_buffer.as_mut_slice());
    layout_buffer
}

/// Copies the row-major data from one matrix, and partially transpose it so
/// memory is structured in row-major order, but in blocks of 16 columns.
unsafe fn copy_16wide_layout( width: usize, height: usize, from: *const f32, to: *mut f32) {
    let mut write_offset = 0;
    let mut i = 0;
    while i < width {
        for j in 0..height {
            let offset = i + j * width;

            let r1 = _mm256_loadu_ps(from.add(offset));
            let r2 = _mm256_loadu_ps(from.add(offset + 8));
            _mm256_storeu_ps(to.add(write_offset), r1);
            _mm256_storeu_ps(to.add(write_offset + 8), r2);

            write_offset += 16;
        }

        i += 16;
    }
}

/// Copies the column-major data from one matrix, and partially transpose it so
/// memory is structured in column-major order, but in blocks of 4 rows.
///
/// * Feel free to change column-major to row-major, it just adjusts the axis.
/// TODO: Maybe I should change that
unsafe fn copy_4wide_layout(width: usize, height: usize, from: *const f32, to: *mut f32) {
    let mut write_offset = 0;
    let mut i = 0;
    while i < width {
        for j in 0..height {
            let offset = i + j * width;

            let r1 = _mm256_loadu_ps(from.add(offset));
            let r2 = _mm256_loadu_ps(from.add(offset + 8));
            _mm256_storeu_ps(to.add(write_offset), r1);
            _mm256_storeu_ps(to.add(write_offset + 8), r2);

            write_offset += 16;
        }

        i += 16;
    }
}


#[cfg(test)]
#[rustfmt::skip]
mod test_prep_step_perf {
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    use super::*;

    const DIMS: usize = 4096;

    #[ignore]
    #[test]
    fn bench_prep_step() {
        let (a, b) = crate::test_utils::get_sample_vectors::<f32>(DIMS * DIMS);

        let mut row_prep = Duration::default();
        let mut col_prep = Duration::default();
        for _ in 0..100 {
            let dims = black_box(DIMS);
            let a = black_box(&a);
            let b = black_box(&b);

            let start = Instant::now();
            let a_buffer = unsafe { prep_row_matrix_component(dims, dims, a) };
            row_prep += start.elapsed();

            let start = Instant::now();
            let b_buffer = unsafe { prep_column_matrix_component(dims, dims, b) };
            col_prep += start.elapsed();

            black_box((a_buffer, b_buffer));
        }
        println!("Row Took: {:?}", row_prep / 100);
        println!("Col Took: {:?}", col_prep / 100);
    }

    #[test]
    fn test_16col_copy_layout() {
        let a = [
            0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0,  11.0,  12.0,  13.0,  14.0,  15.0,
            0.1,  1.1,  2.1,  3.1,  4.1,  5.1,  6.1,  7.1,  8.1,  9.1,  10.1,  11.1,  12.1,  13.1,  14.1,  15.1,

            0.2,  1.2,  2.2,  3.2,  4.2,  5.2,  6.2,  7.2,  8.2,  9.2,  10.2,  11.2,  12.2,  13.2,  14.2,  15.2,
            0.3,  1.3,  2.3,  3.3,  4.3,  5.3,  6.3,  7.3,  8.3,  9.3,  10.3,  11.3,  12.3,  13.3,  14.3,  15.3,

            0.4,  1.4,  2.4,  3.4,  4.4,  5.4,  6.4,  7.4,  8.4,  9.4,  10.4,  11.4,  12.4,  13.4,  14.4,  15.4,
            0.5,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,  9.5,  10.5,  11.5,  12.5,  13.5,  14.5,  15.5,

            0.6,  1.6,  2.6,  3.6,  4.6,  5.6,  6.6,  7.6,  8.6,  9.6,  10.6,  11.6,  12.6,  13.6,  14.6,  15.6,
            0.7,  1.7,  2.7,  3.7,  4.7,  5.7,  6.7,  7.7,  8.7,  9.7,  10.7,  11.7,  12.7,  13.7,  14.7,  15.7,

            0.8,  1.8,  2.8,  3.8,  4.8,  5.8,  6.8,  7.8,  8.8,  9.8,  10.8,  11.8,  12.8,  13.8,  14.8,  15.8,
            0.9,  1.9,  2.9,  3.9,  4.9,  5.9,  6.9,  7.9,  8.9,  9.9,  10.9,  11.9,  12.9,  13.9,  14.9,  15.9,

            0.10, 1.10, 2.10, 3.10, 4.10, 5.10, 6.10, 7.10, 8.10, 9.10, 10.10, 11.10, 12.10, 13.10, 14.10, 15.10,
            0.11, 1.11, 2.11, 3.11, 4.11, 5.11, 6.11, 7.11, 8.11, 9.11, 10.11, 11.11, 12.11, 13.11, 14.11, 15.11,

            0.12, 1.12, 2.12, 3.12, 4.12, 5.12, 6.12, 7.12, 8.12, 9.12, 10.12, 11.12, 12.12, 13.12, 14.12, 15.12,
            0.13, 1.13, 2.13, 3.13, 4.13, 5.13, 6.13, 7.13, 8.13, 9.13, 10.13, 11.13, 12.13, 13.13, 14.13, 15.13,

            0.14, 1.14, 2.14, 3.14, 4.14, 5.14, 6.14, 7.14, 8.14, 9.14, 10.14, 11.14, 12.14, 13.14, 14.14, 15.14,
            0.15, 1.15, 2.15, 3.15, 4.15, 5.15, 6.15, 7.15, 8.15, 9.15, 10.15, 11.15, 12.15, 13.15, 14.15, 15.15,

            0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0,  11.0,  12.0,  13.0,  14.0,  15.0,
            0.1,  1.1,  2.1,  3.1,  4.1,  5.1,  6.1,  7.1,  8.1,  9.1,  10.1,  11.1,  12.1,  13.1,  14.1,  15.1,

            0.2,  1.2,  2.2,  3.2,  4.2,  5.2,  6.2,  7.2,  8.2,  9.2,  10.2,  11.2,  12.2,  13.2,  14.2,  15.2,
            0.3,  1.3,  2.3,  3.3,  4.3,  5.3,  6.3,  7.3,  8.3,  9.3,  10.3,  11.3,  12.3,  13.3,  14.3,  15.3,

            0.4,  1.4,  2.4,  3.4,  4.4,  5.4,  6.4,  7.4,  8.4,  9.4,  10.4,  11.4,  12.4,  13.4,  14.4,  15.4,
            0.5,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,  9.5,  10.5,  11.5,  12.5,  13.5,  14.5,  15.5,

            0.6,  1.6,  2.6,  3.6,  4.6,  5.6,  6.6,  7.6,  8.6,  9.6,  10.6,  11.6,  12.6,  13.6,  14.6,  15.6,
            0.7,  1.7,  2.7,  3.7,  4.7,  5.7,  6.7,  7.7,  8.7,  9.7,  10.7,  11.7,  12.7,  13.7,  14.7,  15.7,

            0.8,  1.8,  2.8,  3.8,  4.8,  5.8,  6.8,  7.8,  8.8,  9.8,  10.8,  11.8,  12.8,  13.8,  14.8,  15.8,
            0.9,  1.9,  2.9,  3.9,  4.9,  5.9,  6.9,  7.9,  8.9,  9.9,  10.9,  11.9,  12.9,  13.9,  14.9,  15.9,

            0.10, 1.10, 2.10, 3.10, 4.10, 5.10, 6.10, 7.10, 8.10, 9.10, 10.10, 11.10, 12.10, 13.10, 14.10, 15.10,
            0.11, 1.11, 2.11, 3.11, 4.11, 5.11, 6.11, 7.11, 8.11, 9.11, 10.11, 11.11, 12.11, 13.11, 14.11, 15.11,

            0.12, 1.12, 2.12, 3.12, 4.12, 5.12, 6.12, 7.12, 8.12, 9.12, 10.12, 11.12, 12.12, 13.12, 14.12, 15.12,
            0.13, 1.13, 2.13, 3.13, 4.13, 5.13, 6.13, 7.13, 8.13, 9.13, 10.13, 11.13, 12.13, 13.13, 14.13, 15.13,

            0.14, 1.14, 2.14, 3.14, 4.14, 5.14, 6.14, 7.14, 8.14, 9.14, 10.14, 11.14, 12.14, 13.14, 14.14, 15.14,
            0.15, 1.15, 2.15, 3.15, 4.15, 5.15, 6.15, 7.15, 8.15, 9.15, 10.15, 11.15, 12.15, 13.15, 14.15, 15.15,
        ];

        let expected = [
            0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0,  11.0,  12.0,  13.0,  14.0,  15.0,
            0.2,  1.2,  2.2,  3.2,  4.2,  5.2,  6.2,  7.2,  8.2,  9.2,  10.2,  11.2,  12.2,  13.2,  14.2,  15.2,
            0.4,  1.4,  2.4,  3.4,  4.4,  5.4,  6.4,  7.4,  8.4,  9.4,  10.4,  11.4,  12.4,  13.4,  14.4,  15.4,
            0.6,  1.6,  2.6,  3.6,  4.6,  5.6,  6.6,  7.6,  8.6,  9.6,  10.6,  11.6,  12.6,  13.6,  14.6,  15.6,
            0.8,  1.8,  2.8,  3.8,  4.8,  5.8,  6.8,  7.8,  8.8,  9.8,  10.8,  11.8,  12.8,  13.8,  14.8,  15.8,
            0.10, 1.10, 2.10, 3.10, 4.10, 5.10, 6.10, 7.10, 8.10, 9.10, 10.10, 11.10, 12.10, 13.10, 14.10, 15.10,
            0.12, 1.12, 2.12, 3.12, 4.12, 5.12, 6.12, 7.12, 8.12, 9.12, 10.12, 11.12, 12.12, 13.12, 14.12, 15.12,
            0.14, 1.14, 2.14, 3.14, 4.14, 5.14, 6.14, 7.14, 8.14, 9.14, 10.14, 11.14, 12.14, 13.14, 14.14, 15.14,
            0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0,  11.0,  12.0,  13.0,  14.0,  15.0,
            0.2,  1.2,  2.2,  3.2,  4.2,  5.2,  6.2,  7.2,  8.2,  9.2,  10.2,  11.2,  12.2,  13.2,  14.2,  15.2,
            0.4,  1.4,  2.4,  3.4,  4.4,  5.4,  6.4,  7.4,  8.4,  9.4,  10.4,  11.4,  12.4,  13.4,  14.4,  15.4,
            0.6,  1.6,  2.6,  3.6,  4.6,  5.6,  6.6,  7.6,  8.6,  9.6,  10.6,  11.6,  12.6,  13.6,  14.6,  15.6,
            0.8,  1.8,  2.8,  3.8,  4.8,  5.8,  6.8,  7.8,  8.8,  9.8,  10.8,  11.8,  12.8,  13.8,  14.8,  15.8,
            0.10, 1.10, 2.10, 3.10, 4.10, 5.10, 6.10, 7.10, 8.10, 9.10, 10.10, 11.10, 12.10, 13.10, 14.10, 15.10,
            0.12, 1.12, 2.12, 3.12, 4.12, 5.12, 6.12, 7.12, 8.12, 9.12, 10.12, 11.12, 12.12, 13.12, 14.12, 15.12,
            0.14, 1.14, 2.14, 3.14, 4.14, 5.14, 6.14, 7.14, 8.14, 9.14, 10.14, 11.14, 12.14, 13.14, 14.14, 15.14,

            0.1,  1.1,  2.1,  3.1,  4.1,  5.1,  6.1,  7.1,  8.1,  9.1,  10.1,  11.1,  12.1,  13.1,  14.1,  15.1,
            0.3,  1.3,  2.3,  3.3,  4.3,  5.3,  6.3,  7.3,  8.3,  9.3,  10.3,  11.3,  12.3,  13.3,  14.3,  15.3,
            0.5,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,  9.5,  10.5,  11.5,  12.5,  13.5,  14.5,  15.5,
            0.7,  1.7,  2.7,  3.7,  4.7,  5.7,  6.7,  7.7,  8.7,  9.7,  10.7,  11.7,  12.7,  13.7,  14.7,  15.7,
            0.9,  1.9,  2.9,  3.9,  4.9,  5.9,  6.9,  7.9,  8.9,  9.9,  10.9,  11.9,  12.9,  13.9,  14.9,  15.9,
            0.11, 1.11, 2.11, 3.11, 4.11, 5.11, 6.11, 7.11, 8.11, 9.11, 10.11, 11.11, 12.11, 13.11, 14.11, 15.11,
            0.13, 1.13, 2.13, 3.13, 4.13, 5.13, 6.13, 7.13, 8.13, 9.13, 10.13, 11.13, 12.13, 13.13, 14.13, 15.13,
            0.15, 1.15, 2.15, 3.15, 4.15, 5.15, 6.15, 7.15, 8.15, 9.15, 10.15, 11.15, 12.15, 13.15, 14.15, 15.15,
            0.1,  1.1,  2.1,  3.1,  4.1,  5.1,  6.1,  7.1,  8.1,  9.1,  10.1,  11.1,  12.1,  13.1,  14.1,  15.1,
            0.3,  1.3,  2.3,  3.3,  4.3,  5.3,  6.3,  7.3,  8.3,  9.3,  10.3,  11.3,  12.3,  13.3,  14.3,  15.3,
            0.5,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,  9.5,  10.5,  11.5,  12.5,  13.5,  14.5,  15.5,
            0.7,  1.7,  2.7,  3.7,  4.7,  5.7,  6.7,  7.7,  8.7,  9.7,  10.7,  11.7,  12.7,  13.7,  14.7,  15.7,
            0.9,  1.9,  2.9,  3.9,  4.9,  5.9,  6.9,  7.9,  8.9,  9.9,  10.9,  11.9,  12.9,  13.9,  14.9,  15.9,
            0.11, 1.11, 2.11, 3.11, 4.11, 5.11, 6.11, 7.11, 8.11, 9.11, 10.11, 11.11, 12.11, 13.11, 14.11, 15.11,
            0.13, 1.13, 2.13, 3.13, 4.13, 5.13, 6.13, 7.13, 8.13, 9.13, 10.13, 11.13, 12.13, 13.13, 14.13, 15.13,
            0.15, 1.15, 2.15, 3.15, 4.15, 5.15, 6.15, 7.15, 8.15, 9.15, 10.15, 11.15, 12.15, 13.15, 14.15, 15.15,
        ];

        let mut result = [999.0; 512];
        unsafe { copy_16wide_layout(32, 16, a.as_ptr(), result.as_mut_ptr()) };

        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_4x16_matmul_kernel() {
        let a = [
            0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0,  11.0,  12.0,  13.0,  14.0,  15.0,
            0.1,  1.1,  2.1,  3.1,  4.1,  5.1,  6.1,  7.1,  8.1,  9.1,  10.1,  11.1,  12.1,  13.1,  14.1,  15.1,
            0.2,  1.2,  2.2,  3.2,  4.2,  5.2,  6.2,  7.2,  8.2,  9.2,  10.2,  11.2,  12.2,  13.2,  14.2,  15.2,
            0.3,  1.3,  2.3,  3.3,  4.3,  5.3,  6.3,  7.3,  8.3,  9.3,  10.3,  11.3,  12.3,  13.3,  14.3,  15.3,
            0.4,  1.4,  2.4,  3.4,  4.4,  5.4,  6.4,  7.4,  8.4,  9.4,  10.4,  11.4,  12.4,  13.4,  14.4,  15.4,
            0.5,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,  9.5,  10.5,  11.5,  12.5,  13.5,  14.5,  15.5,
            0.6,  1.6,  2.6,  3.6,  4.6,  5.6,  6.6,  7.6,  8.6,  9.6,  10.6,  11.6,  12.6,  13.6,  14.6,  15.6,
            0.7,  1.7,  2.7,  3.7,  4.7,  5.7,  6.7,  7.7,  8.7,  9.7,  10.7,  11.7,  12.7,  13.7,  14.7,  15.7,
            0.8,  1.8,  2.8,  3.8,  4.8,  5.8,  6.8,  7.8,  8.8,  9.8,  10.8,  11.8,  12.8,  13.8,  14.8,  15.8,
            0.9,  1.9,  2.9,  3.9,  4.9,  5.9,  6.9,  7.9,  8.9,  9.9,  10.9,  11.9,  12.9,  13.9,  14.9,  15.9,
            0.10, 1.10, 2.10, 3.10, 4.10, 5.10, 6.10, 7.10, 8.10, 9.10, 10.10, 11.10, 12.10, 13.10, 14.10, 15.10,
            0.11, 1.11, 2.11, 3.11, 4.11, 5.11, 6.11, 7.11, 8.11, 9.11, 10.11, 11.11, 12.11, 13.11, 14.11, 15.11,
            0.12, 1.12, 2.12, 3.12, 4.12, 5.12, 6.12, 7.12, 8.12, 9.12, 10.12, 11.12, 12.12, 13.12, 14.12, 15.12,
            0.13, 1.13, 2.13, 3.13, 4.13, 5.13, 6.13, 7.13, 8.13, 9.13, 10.13, 11.13, 12.13, 13.13, 14.13, 15.13,
            0.14, 1.14, 2.14, 3.14, 4.14, 5.14, 6.14, 7.14, 8.14, 9.14, 10.14, 11.14, 12.14, 13.14, 14.14, 15.14,
            0.15, 1.15, 2.15, 3.15, 4.15, 5.15, 6.15, 7.15, 8.15, 9.15, 10.15, 11.15, 12.15, 13.15, 14.15, 15.15,
        ];

        let b = [
            0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0,  11.0,  12.0,  13.0,  14.0,  15.0,
            0.1,  1.1,  2.1,  3.1,  4.1,  5.1,  6.1,  7.1,  8.1,  9.1,  10.1,  11.1,  12.1,  13.1,  14.1,  15.1,
            0.2,  1.2,  2.2,  3.2,  4.2,  5.2,  6.2,  7.2,  8.2,  9.2,  10.2,  11.2,  12.2,  13.2,  14.2,  15.2,
            0.3,  1.3,  2.3,  3.3,  4.3,  5.3,  6.3,  7.3,  8.3,  9.3,  10.3,  11.3,  12.3,  13.3,  14.3,  15.3,
            0.4,  1.4,  2.4,  3.4,  4.4,  5.4,  6.4,  7.4,  8.4,  9.4,  10.4,  11.4,  12.4,  13.4,  14.4,  15.4,
            0.5,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,  9.5,  10.5,  11.5,  12.5,  13.5,  14.5,  15.5,
            0.6,  1.6,  2.6,  3.6,  4.6,  5.6,  6.6,  7.6,  8.6,  9.6,  10.6,  11.6,  12.6,  13.6,  14.6,  15.6,
            0.7,  1.7,  2.7,  3.7,  4.7,  5.7,  6.7,  7.7,  8.7,  9.7,  10.7,  11.7,  12.7,  13.7,  14.7,  15.7,
            0.8,  1.8,  2.8,  3.8,  4.8,  5.8,  6.8,  7.8,  8.8,  9.8,  10.8,  11.8,  12.8,  13.8,  14.8,  15.8,
            0.9,  1.9,  2.9,  3.9,  4.9,  5.9,  6.9,  7.9,  8.9,  9.9,  10.9,  11.9,  12.9,  13.9,  14.9,  15.9,
            0.10, 1.10, 2.10, 3.10, 4.10, 5.10, 6.10, 7.10, 8.10, 9.10, 10.10, 11.10, 12.10, 13.10, 14.10, 15.10,
            0.11, 1.11, 2.11, 3.11, 4.11, 5.11, 6.11, 7.11, 8.11, 9.11, 10.11, 11.11, 12.11, 13.11, 14.11, 15.11,
            0.12, 1.12, 2.12, 3.12, 4.12, 5.12, 6.12, 7.12, 8.12, 9.12, 10.12, 11.12, 12.12, 13.12, 14.12, 15.12,
            0.13, 1.13, 2.13, 3.13, 4.13, 5.13, 6.13, 7.13, 8.13, 9.13, 10.13, 11.13, 12.13, 13.13, 14.13, 15.13,
            0.14, 1.14, 2.14, 3.14, 4.14, 5.14, 6.14, 7.14, 8.14, 9.14, 10.14, 11.14, 12.14, 13.14, 14.14, 15.14,
            0.15, 1.15, 2.15, 3.15, 4.15, 5.15, 6.15, 7.15, 8.15, 9.15, 10.15, 11.15, 12.15, 13.15, 14.15, 15.15,
        ];

        let mut result = vec![0.0; 16 * 16];

        unsafe {
            matmul(16, 16, &a, 16, 16, &b, &mut result);
        };

        for i in 0..16 {
            let slice = &result[i*16..16 + i*16];
            println!("{slice:?}")        ;
        }
    }
}