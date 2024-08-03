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

    bencher.bench_local(|| {
        let dims = black_box(DIMS);
        let l1 = black_box(&l1);
        let l2 = black_box(&l2);

        let mut result = black_box(result.clone());

        unsafe {
            cfavml_gemm::danger::f32_avx2fma::matmul(
                dims,
                dims,
                l1,
                dims,
                dims,
                l2,
                &mut result,
            );
        }

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



#[cfg_attr(
    not(debug_assertions),
    divan::bench(
        sample_count = 10,
        sample_size = 1,
        counters = [divan::counter::ItemsCount::new(FLOP)],
    )
)]
fn bench_gemm_faer(bencher: Bencher) {
    let (l1, l2) = utils::get_sample_vectors::<f32>(DIMS * DIMS);
    let l1 = faer::mat::from_row_major_slice(&l1, DIMS, DIMS);
    let l2 = faer::mat::from_row_major_slice(&l2, DIMS, DIMS);
    let result = faer::mat::Mat::zeros(DIMS, DIMS);

    bencher.bench_local(|| {
        let l1 = black_box(&l1);
        let l2 = black_box(&l2);
        let mut result = black_box(result.clone());

        faer::linalg::matmul::matmul(
            &mut result,
            &l1,
            &l2,
            None,
            1.0,
            faer::Parallelism::None,
        );

        result
    });
}