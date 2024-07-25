use std::hint::black_box;

use cfavml_gemm::transpose::transpose_matrix;
use cfavml_utils::aligned_buffer::AlignedBuffer;
use divan::Bencher;
use rand::distributions::{Distribution, Standard};

mod utils;

const DIMS: usize = 4096;

fn main() {
    divan::main()
}

#[cfg_attr(
    not(debug_assertions),
    divan::bench(
        types = [f32, f64],
        counters = [divan::counter::ItemsCount::new(DIMS * DIMS)],
    )
)]
fn bench_transpose_cfavml<T>(bencher: Bencher)
where
    T: Copy + 'static,
    Standard: Distribution<T>,
{
    let (l1, _) = utils::get_sample_vectors::<T>(DIMS * DIMS);
    let mut buffer: AlignedBuffer<T> = unsafe { AlignedBuffer::zeroed(DIMS * DIMS) };

    bencher.bench_local(|| {
        let buffer = black_box(buffer.as_mut_slice());
        let data = black_box(&l1);
        let dims = black_box(DIMS);

        transpose_matrix(dims, dims, data, buffer)
    });
}

#[cfg_attr(
    not(debug_assertions),
    divan::bench(
        types = [f32, f64],
        counters = [divan::counter::ItemsCount::new(DIMS * DIMS)],
    )
)]
fn bench_transpose_basic<T>(bencher: Bencher)
where
    T: Copy + 'static,
    Standard: Distribution<T>,
{
    let (l1, _) = utils::get_sample_vectors::<T>(DIMS * DIMS);
    let mut buffer: AlignedBuffer<T> = unsafe { AlignedBuffer::zeroed(DIMS * DIMS) };

    bencher.bench_local(|| {
        let buffer = black_box(buffer.as_mut_slice());
        let data = black_box(&l1);
        let dims = black_box(DIMS);

        unsafe {
            for i in 0..dims {
                for j in 0..dims {
                    *buffer.get_unchecked_mut(i * dims + j) =
                        *data.get_unchecked(j * dims + i);
                }
            }
        }
    });
}
