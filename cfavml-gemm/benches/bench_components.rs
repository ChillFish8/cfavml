use std::arch::x86_64::_mm256_loadu_ps;
use std::hint::black_box;
use divan::Bencher;
use cfavml_utils::aligned_buffer::AlignedBuffer;

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
/// Performs a copy of memory from the row-major matrix into an aligned buffer of
/// DIMS * 8 in side, this tries to mimic the overhead during the GEMM operation
/// within the kernel.
fn bench_row_major_to_columns_load_aligned(bencher: Bencher) {
    let (l1, _) = utils::get_sample_vectors::<f32>(DIMS * DIMS);
    let mut buffer: AlignedBuffer<f32> = unsafe { AlignedBuffer::zeroed(DIMS * 8) };

    bencher.bench_local(|| {
        let buffer = black_box(buffer.as_mut_slice());
        let data = black_box(&l1);
        let dims = black_box(DIMS);

        let mut x_offset = 0;
        while x_offset < dims {

            let mut cursor = 0;
            let mut y_offset = 0;
            while y_offset < dims {
                let start = (dims * y_offset) + x_offset;

                let slice = unsafe { data.get_unchecked(start..start + 8) };
                unsafe { buffer.get_unchecked_mut(cursor..cursor + 8).copy_from_slice(slice) };

                cursor += slice.len();

                y_offset += 1;
            }

            x_offset += 8;
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
fn bench_row_major_to_column_major_transpose(bencher: Bencher) {
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
/// Performs a copy of memory from the row-major matrix into an aligned buffer of
/// DIMS * 8 in side, this tries to mimic the overhead during the GEMM operation
/// within the kernel.
fn bench_row_major_to_rows_load_aligned(bencher: Bencher) {
    let (l1, _) = utils::get_sample_vectors::<f32>(DIMS * DIMS);
    let mut buffer: AlignedBuffer<f32> = unsafe { AlignedBuffer::zeroed(DIMS * 8) };

    bencher.bench_local(|| {
        let buffer = black_box(buffer.as_mut_slice());
        let data = black_box(&l1);
        let dims = black_box(DIMS);
        let block_size =  dims * 8;

        let mut offset = 0;
        while offset < (dims * dims) {
            let slice = unsafe { data.get_unchecked(offset..offset + block_size) };
            unsafe { buffer.copy_from_slice(slice) };

            offset += block_size;
        }
    });
}