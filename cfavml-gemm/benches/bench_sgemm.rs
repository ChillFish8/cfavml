use std::hint::black_box;
use divan::{Bencher, counter::ItemsCount};
use faer::Parallelism;
use ndarray::{Array2, LinalgScalar};

mod utils;

fn main() {
    divan::main();
}


#[cfg_attr(
    not(debug_assertions),
    divan::bench(
        sample_size = 1,
        sample_count = 10,
        counters = [divan::counter::ItemsCount::new(4096usize * 4096 * 2 * 4096)]
    )
)]
fn ndarray_matmul(bencher: Bencher) {
    let (a, b) = utils::get_sample_vectors::<f32>(4096 * 4096);
    let a = Array2::from_shape_vec((4096, 4096), a).unwrap();
    let b = Array2::from_shape_vec((4096, 4096), b).unwrap();
    let mut c = Array2::zeros((4096, 4096));

    bencher.bench_local(|| ndarray::linalg::general_mat_mul(0.0, black_box(&a), black_box(&b), 1.0, black_box(&mut c)));
}

#[cfg_attr(
    not(debug_assertions),
    divan::bench(
        sample_size = 1,
        sample_count = 10,
        counters = [divan::counter::ItemsCount::new(4096usize * 4096 * 2 * 4096)]
    )
)]
fn faer_matmul(bencher: Bencher) {
    let (a, b) = utils::get_sample_vectors::<f32>(4096 * 4096);
    let a = faer::mat::from_row_major_slice(&a, 4096, 4096);
    let b = faer::mat::from_row_major_slice(&b, 4096, 4096);
    let mut c = faer::Mat::zeros(4096, 4096);

    bencher.bench_local(|| faer::linalg::matmul::matmul(
        black_box(&mut c),
        black_box(&a),
        black_box(&b),
        None,
        1.0,
        Parallelism::None,
    ));
}
