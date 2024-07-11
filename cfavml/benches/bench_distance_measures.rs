#[cfg(unix)]
extern crate blas_src;

use std::hint::black_box;

use cfavml::danger::*;
use cfavml::math::*;
use divan::Bencher;
use ndarray::ArrayView1;

mod utils;

const DIMS: usize = 1341;

fn main() {
    divan::main();
}

#[cfg_attr(
    not(debug_assertions),
    divan::bench_group(sample_count = 500, sample_size = 2500, threads = false)
)]
mod dot_product_x1341 {
    use super::*;

    #[cfg_attr(not(debug_assertions), divan::bench)]
    fn ndarray_f32(bencher: Bencher) {
        let (l1, l2) = utils::get_sample_vectors::<f32>(DIMS);
        let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();
        let l2_view = ArrayView1::from_shape((l2.len(),), &l2).unwrap();

        bencher.bench_local(|| {
            let l1_view = black_box(l1_view);
            let l2_view = black_box(l2_view);

            l1_view.dot(&l2_view)
        });
    }

    #[cfg_attr(not(debug_assertions), divan::bench)]
    fn ndarray_f64(bencher: Bencher) {
        let (l1, l2) = utils::get_sample_vectors::<f64>(DIMS);
        let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();
        let l2_view = ArrayView1::from_shape((l2.len(),), &l2).unwrap();

        bencher.bench_local(|| {
            let l1_view = black_box(l1_view);
            let l2_view = black_box(l2_view);

            l1_view.dot(&l2_view)
        });
    }

    #[cfg_attr(not(debug_assertions), divan::bench)]
    fn simsimd_f32(bencher: Bencher) {
        let (l1, l2) = utils::get_sample_vectors::<f32>(DIMS);

        bencher.bench_local(|| {
            let l1_view = black_box(&l1);
            let l2_view = black_box(&l2);

            simsimd::SpatialSimilarity::dot(l1_view, l2_view).unwrap_or_default()
        });
    }

    #[cfg_attr(not(debug_assertions), divan::bench)]
    fn simsimd_f64(bencher: Bencher) {
        let (l1, l2) = utils::get_sample_vectors::<f64>(DIMS);

        bencher.bench_local(|| {
            let l1_view = black_box(&l1);
            let l2_view = black_box(&l2);

            simsimd::SpatialSimilarity::dot(l1_view, l2_view).unwrap_or_default()
        });
    }

    macro_rules! define_cfavml_variants {
        (
            $t:ident,
            $name:ident,
            $op:expr
        ) => {
            #[cfg_attr(not(debug_assertions), divan::bench)]
            fn $name(bencher: Bencher) {
                let (l1, l2) = utils::get_sample_vectors::<$t>(DIMS);

                bencher.bench_local(|| {
                    let l1_view = black_box(&l1);
                    let l2_view = black_box(&l2);
                    unsafe { $op(l1_view, l2_view) }
                });
            }
        };
    }

    define_cfavml_variants!(f32, cfavml_fallback_nofma_f32, f32_xany_fallback_nofma_dot);
    define_cfavml_variants!(f64, cfavml_fallback_nofma_f64, f64_xany_fallback_nofma_dot);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    define_cfavml_variants!(f32, cfavml_avx2_nofma_f32, f32_xany_avx2_nofma_dot);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    define_cfavml_variants!(f64, cfavml_avx2_nofma_f64, f64_xany_avx2_nofma_dot);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    define_cfavml_variants!(f32, cfavml_avx2_fma_f32, f32_xany_avx2_fma_dot);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    define_cfavml_variants!(f64, cfavml_avx2_fma_f64, f64_xany_avx2_fma_dot);

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "benchmark-avx512"
    ))]
    define_cfavml_variants!(f32, cfavml_avx512_fma_f32, f32_xany_avx512_fma_dot);
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "benchmark-avx512"
    ))]
    define_cfavml_variants!(f64, cfavml_avx512_fma_f64, f64_xany_avx512_fma_dot);

    #[cfg(target_arch = "aarch64")]
    define_cfavml_variants!(f32, cfavml_neon_fma_f32, f32_xany_neon_fma_dot);
    #[cfg(target_arch = "aarch64")]
    define_cfavml_variants!(f64, cfavml_neon_fma_f64, f64_xany_neon_fma_dot);
}

#[cfg_attr(
    not(debug_assertions),
    divan::bench_group(sample_count = 500, sample_size = 2500, threads = false)
)]
mod cosine_x1341 {
    use super::*;

    #[cfg_attr(not(debug_assertions), divan::bench)]
    fn ndarray_f32(bencher: Bencher) {
        let (l1, l2) = utils::get_sample_vectors::<f32>(DIMS);
        let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();
        let l2_view = ArrayView1::from_shape((l2.len(),), &l2).unwrap();

        bencher.bench_local(|| {
            let l1_view = black_box(l1_view);
            let l2_view = black_box(l2_view);

            let norm_l1 = l1_view.product();
            let norm_l2 = l2_view.product();
            let dot = l1_view.dot(&l2_view);
            utils::cosine::<_, AutoMath>(dot, norm_l1, norm_l2)
        });
    }

    #[cfg_attr(not(debug_assertions), divan::bench)]
    fn ndarray_f64(bencher: Bencher) {
        let (l1, l2) = utils::get_sample_vectors::<f64>(DIMS);
        let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();
        let l2_view = ArrayView1::from_shape((l2.len(),), &l2).unwrap();

        bencher.bench_local(|| {
            let l1_view = black_box(l1_view);
            let l2_view = black_box(l2_view);

            let norm_l1 = l1_view.product();
            let norm_l2 = l2_view.product();
            let dot = l1_view.dot(&l2_view);
            utils::cosine::<_, AutoMath>(dot, norm_l1, norm_l2)
        });
    }

    #[cfg_attr(not(debug_assertions), divan::bench)]
    fn simsimd_f32(bencher: Bencher) {
        let (l1, l2) = utils::get_sample_vectors::<f32>(DIMS);

        bencher.bench_local(|| {
            let l1_view = black_box(&l1);
            let l2_view = black_box(&l2);

            simsimd::SpatialSimilarity::cosine(l1_view, l2_view).unwrap_or_default()
        });
    }

    #[cfg_attr(not(debug_assertions), divan::bench)]
    fn simsimd_f64(bencher: Bencher) {
        let (l1, l2) = utils::get_sample_vectors::<f64>(DIMS);

        bencher.bench_local(|| {
            let l1_view = black_box(&l1);
            let l2_view = black_box(&l2);

            simsimd::SpatialSimilarity::cosine(l1_view, l2_view).unwrap_or_default()
        });
    }

    macro_rules! define_cfavml_variants {
        (
            $t:ident,
            $name:ident,
            $op:expr
        ) => {
            #[cfg_attr(not(debug_assertions), divan::bench)]
            fn $name(bencher: Bencher) {
                let (l1, l2) = utils::get_sample_vectors::<$t>(DIMS);

                bencher.bench_local(|| {
                    let l1_view = black_box(&l1);
                    let l2_view = black_box(&l2);
                    unsafe { $op(l1_view, l2_view) }
                });
            }
        };
    }

    define_cfavml_variants!(
        f32,
        cfavml_fallback_nofma_f32,
        f32_xany_fallback_nofma_cosine
    );
    define_cfavml_variants!(
        f64,
        cfavml_fallback_nofma_f64,
        f64_xany_fallback_nofma_cosine
    );

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    define_cfavml_variants!(f32, cfavml_avx2_nofma_f32, f32_xany_avx2_nofma_cosine);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    define_cfavml_variants!(f64, cfavml_avx2_nofma_f64, f64_xany_avx2_nofma_cosine);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    define_cfavml_variants!(f32, cfavml_avx2_fma_f32, f32_xany_avx2_fma_cosine);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    define_cfavml_variants!(f64, cfavml_avx2_fma_f64, f64_xany_avx2_fma_cosine);

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "benchmark-avx512"
    ))]
    define_cfavml_variants!(f32, cfavml_avx512_fma_f32, f32_xany_avx512_fma_cosine);
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "benchmark-avx512"
    ))]
    define_cfavml_variants!(f64, cfavml_avx512_fma_f64, f64_xany_avx512_fma_cosine);

    #[cfg(target_arch = "aarch64")]
    define_cfavml_variants!(f32, cfavml_neon_fma_f32, f32_xany_neon_fma_cosine);
    #[cfg(target_arch = "aarch64")]
    define_cfavml_variants!(f64, cfavml_neon_fma_f64, f64_xany_neon_fma_cosine);
}

#[cfg_attr(
    not(debug_assertions),
    divan::bench_group(sample_count = 500, sample_size = 2500, threads = false)
)]
mod euclidean_x1341 {
    use std::ops::Sub;

    use super::*;

    #[cfg_attr(not(debug_assertions), divan::bench)]
    fn ndarray_f32(bencher: Bencher) {
        let (l1, l2) = utils::get_sample_vectors::<f32>(DIMS);
        let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();
        let l2_view = ArrayView1::from_shape((l2.len(),), &l2).unwrap();

        bencher.bench_local(|| {
            let l1_view = black_box(l1_view);
            let l2_view = black_box(l2_view);

            let diff = l1_view.sub(&l2_view);
            diff.product()
        });
    }

    #[cfg_attr(not(debug_assertions), divan::bench)]
    fn ndarray_f64(bencher: Bencher) {
        let (l1, l2) = utils::get_sample_vectors::<f64>(DIMS);
        let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();
        let l2_view = ArrayView1::from_shape((l2.len(),), &l2).unwrap();

        bencher.bench_local(|| {
            let l1_view = black_box(l1_view);
            let l2_view = black_box(l2_view);

            let diff = l1_view.sub(&l2_view);
            diff.product()
        });
    }

    #[cfg_attr(not(debug_assertions), divan::bench)]
    fn simsimd_f32(bencher: Bencher) {
        let (l1, l2) = utils::get_sample_vectors::<f32>(DIMS);

        bencher.bench_local(|| {
            let l1_view = black_box(&l1);
            let l2_view = black_box(&l2);

            simsimd::SpatialSimilarity::sqeuclidean(l1_view, l2_view).unwrap_or_default()
        });
    }

    #[cfg_attr(not(debug_assertions), divan::bench)]
    fn simsimd_f64(bencher: Bencher) {
        let (l1, l2) = utils::get_sample_vectors::<f64>(DIMS);

        bencher.bench_local(|| {
            let l1_view = black_box(&l1);
            let l2_view = black_box(&l2);

            simsimd::SpatialSimilarity::sqeuclidean(l1_view, l2_view).unwrap_or_default()
        });
    }

    macro_rules! define_cfavml_variants {
        (
            $t:ident,
            $name:ident,
            $op:expr
        ) => {
            #[cfg_attr(not(debug_assertions), divan::bench)]
            fn $name(bencher: Bencher) {
                let (l1, l2) = utils::get_sample_vectors::<$t>(DIMS);

                bencher.bench_local(|| {
                    let l1_view = black_box(&l1);
                    let l2_view = black_box(&l2);
                    unsafe { $op(l1_view, l2_view) }
                });
            }
        };
    }

    define_cfavml_variants!(
        f32,
        cfavml_fallback_nofma_f32,
        f32_xany_fallback_nofma_squared_euclidean
    );
    define_cfavml_variants!(
        f64,
        cfavml_fallback_nofma_f64,
        f64_xany_fallback_nofma_squared_euclidean
    );

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    define_cfavml_variants!(
        f32,
        cfavml_avx2_nofma_f32,
        f32_xany_avx2_nofma_squared_euclidean
    );
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    define_cfavml_variants!(
        f64,
        cfavml_avx2_nofma_f64,
        f64_xany_avx2_nofma_squared_euclidean
    );
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    define_cfavml_variants!(
        f32,
        cfavml_avx2_fma_f32,
        f32_xany_avx2_fma_squared_euclidean
    );
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    define_cfavml_variants!(
        f64,
        cfavml_avx2_fma_f64,
        f64_xany_avx2_fma_squared_euclidean
    );

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "benchmark-avx512"
    ))]
    define_cfavml_variants!(
        f32,
        cfavml_avx512_fma_f32,
        f32_xany_avx512_fma_squared_euclidean
    );
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "benchmark-avx512"
    ))]
    define_cfavml_variants!(
        f64,
        cfavml_avx512_fma_f64,
        f64_xany_avx512_fma_squared_euclidean
    );

    #[cfg(target_arch = "aarch64")]
    define_cfavml_variants!(
        f32,
        cfavml_neon_fma_f32,
        f32_xany_neon_fma_squared_euclidean
    );
    #[cfg(target_arch = "aarch64")]
    define_cfavml_variants!(
        f64,
        cfavml_neon_fma_f64,
        f64_xany_neon_fma_squared_euclidean
    );
}
