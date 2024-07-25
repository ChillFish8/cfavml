use std::hint::black_box;
use divan::Bencher;

use cfavml_utils::aligned_buffer::AlignedBuffer;
use cfavml_gemm::transpose::{
    transpose_matrix,
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
