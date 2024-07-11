use std::hint::black_box;

use divan::Bencher;

mod utils;

fn main() {
    divan::main();
}

const DIMS: usize = 2048;

#[cfg_attr(not(debug_assertions), divan::bench_group(threads = false))]
mod gemm_x4064 {
    use ndarray::linalg::general_mat_mul;
    use ndarray::{Array2, ArrayView2};

    use super::*;

    #[cfg_attr(not(debug_assertions), divan::bench)]
    fn ndarray_f32(bencher: Bencher) {
        let (l1, l2) = utils::get_sample_vectors::<f32>(DIMS * DIMS);
        let l1_view = ArrayView2::from_shape((DIMS, DIMS), &l1).unwrap();
        let l2_view = ArrayView2::from_shape((DIMS, DIMS), &l2).unwrap();
        let mut l3_view = Array2::from_elem((DIMS, DIMS), 0.0);

        bencher.bench_local(|| {
            let l1_view = black_box(l1_view);
            let l2_view = black_box(l2_view);
            let l3_view = black_box(&mut l3_view);

            general_mat_mul(0.0, &l1_view, &l2_view, 0.0, l3_view)
        });
    }

    #[cfg_attr(not(debug_assertions), divan::bench)]
    fn ndarray_f64(bencher: Bencher) {
        let (l1, l2) = utils::get_sample_vectors::<f64>(DIMS * DIMS);
        let l1_view = ArrayView2::from_shape((DIMS, DIMS), &l1).unwrap();
        let l2_view = ArrayView2::from_shape((DIMS, DIMS), &l2).unwrap();
        let mut l3_view = Array2::from_elem((DIMS, DIMS), 0.0);

        bencher.bench_local(|| {
            let l1_view = black_box(l1_view);
            let l2_view = black_box(l2_view);
            let l3_view = black_box(&mut l3_view);

            general_mat_mul(0.0, &l1_view, &l2_view, 1.0, l3_view)
        });
    }
}
