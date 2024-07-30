use std::hint::black_box;

use divan::Bencher;
use cfavml::danger::*;

mod utils;

const DIMS: usize = 4096;
const FLOP: usize = DIMS * DIMS * 2 * DIMS;

fn main() {
    divan::main()
}


#[cfg_attr(
    not(debug_assertions),
    divan::bench(
        sample_count = 10,
        sample_size = 1,
        counters = [divan::counter::ItemsCount::new(FLOP)],
    )
)]
fn bench_gemm_cfavml(bencher: Bencher) {
    let (l1, l2) = utils::get_sample_vectors::<f32>(DIMS * DIMS);
    let result = vec![0.0; DIMS * DIMS];
    let mut kernel = cfavml_gemm::GenericMatrixKernel::<f32, Avx2>::allocate(DIMS);

    bencher.bench_local(|| {
        let kernel = black_box(&mut kernel);
        let dims = black_box(DIMS);
        let l1 = black_box(&l1);
        let l2 = black_box(&l2);

        let mut result = black_box(result.clone());

        unsafe { kernel.dot_matrix(dims, dims, l1, dims, dims, l2,  &mut result) };

        result
    });
}


#[cfg_attr(
    not(debug_assertions),
    divan::bench(
        sample_count = 10,
        sample_size = 1,
        counters = [divan::counter::ItemsCount::new(FLOP)],
    )
)]
fn bench_gemm_ndarray(bencher: Bencher) {
    let (l1, l2) = utils::get_sample_vectors::<f32>(DIMS * DIMS);
    let l1 = ndarray::Array2::from_shape_vec((DIMS, DIMS), l1).unwrap();
    let l2 = ndarray::Array2::from_shape_vec((DIMS, DIMS), l2).unwrap();
    let result = ndarray::Array2::zeros((DIMS, DIMS));

    bencher.bench_local(|| {
        let l1 = black_box(&l1);
        let l2 = black_box(&l2);
        let mut result = black_box(result.clone());

        ndarray::linalg::general_mat_mul(
            1.0,
            l1,
            l2,
            0.0,
            &mut result,
        );

        result
    });
}