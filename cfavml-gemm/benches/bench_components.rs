use std::arch::x86_64::_mm256_loadu_ps;
use std::hint::black_box;
use divan::Bencher;

use cfavml_utils::aligned_buffer::AlignedBuffer;
use cfavml_gemm::manipulation::{
    transpose_matrix,
    select_row_major_column_row_blocks_sub_buffer,
    select_row_major_row_wise_sub_buffer,
};

mod utils;


const DIMS: usize = 4096;

fn main() {
    divan::main()
}

#[cfg_attr(
    not(debug_assertions),
    divan::bench(
        counters = [divan::counter::ItemsCount::new(DIMS * DIMS)],
    )
)]
fn bench_transpose_cfavml(bencher: Bencher) {
    let (l1, _) = utils::get_sample_vectors::<f32>(DIMS * DIMS);
    let mut buffer: AlignedBuffer<f32> = unsafe { AlignedBuffer::zeroed(DIMS * DIMS) };

    bencher.bench_local(|| {
        let buffer = black_box(buffer.as_mut_slice());
        let data = black_box(&l1);
        let dims = black_box(DIMS);

        unsafe { transpose_matrix(dims, dims, data, buffer) }
    });
}

#[cfg_attr(
    not(debug_assertions),
    divan::bench(
        counters = [divan::counter::ItemsCount::new(DIMS * DIMS)],
    )
)]
fn bench_transpose_basic(bencher: Bencher) {
    let (l1, _) = utils::get_sample_vectors::<f32>(DIMS * DIMS);
    let mut buffer: AlignedBuffer<f32> = unsafe { AlignedBuffer::zeroed(DIMS * DIMS) };

    bencher.bench_local(|| {
        let buffer = black_box(buffer.as_mut_slice());
        let data = black_box(&l1);
        let dims = black_box(DIMS);

        unsafe {
            for i in 0..dims {
                for j in 0..dims {
                    *buffer.get_unchecked_mut(i * dims + j) = *data.get_unchecked(j * dims + i);
                }
            }
        }
    });
}


#[cfg_attr(
    not(debug_assertions),
    divan::bench(
        counters = [divan::counter::ItemsCount::new(DIMS * DIMS)],
    )
)]
/// Performs a copy of memory from the row-major matrix into an column-major matrix.
fn bench_select_row_major_column_row_blocks_sub_buffer(bencher: Bencher) {
    let (l1, _) = utils::get_sample_vectors::<f32>(DIMS * DIMS);
    let mut buffer: AlignedBuffer<f32> = unsafe { AlignedBuffer::zeroed(DIMS * 8) };

    bencher.bench_local(|| {
        let buffer = black_box(buffer.as_mut_slice());
        let data = black_box(&l1);
        let dims = black_box(DIMS);

        for offset in 0..(dims / 8) {
            unsafe { select_row_major_column_row_blocks_sub_buffer::<_, 8>(dims, offset, data, buffer) }
        }
    });
}

#[cfg_attr(
    not(debug_assertions),
    divan::bench(
        counters = [divan::counter::ItemsCount::new(DIMS * DIMS)],
    )
)]
/// Performs a copy of memory from the row-major matrix into an aligned buffer of
/// DIMS * 8 in side, this tries to mimic the overhead during the GEMM operation
/// within the kernel.
fn bench_select_row_major_row_wise_sub_buffer(bencher: Bencher) {
    let (l1, _) = utils::get_sample_vectors::<f32>(DIMS * DIMS);
    let mut buffer: AlignedBuffer<f32> = unsafe { AlignedBuffer::zeroed(DIMS * 8) };

    bencher.bench_local(|| {
        let buffer = black_box(buffer.as_mut_slice());
        let data = black_box(&l1);
        let dims = black_box(DIMS);

        for offset in 0..(dims / 8) {
            unsafe { select_row_major_row_wise_sub_buffer::<_, 8>(dims, offset, data, buffer) }
        }
    });
}