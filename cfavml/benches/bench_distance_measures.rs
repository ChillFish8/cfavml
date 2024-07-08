#[cfg(unix)]
extern crate blas_src;

use std::hint::black_box;
use std::ops::Sub;
use std::time::Duration;

use cfavml::danger::*;
use cfavml::math::AutoMath;
use criterion::{criterion_group, criterion_main, Criterion};

mod utils;

macro_rules! benchmark_distance_measure {
    (
        $name:expr,
        x1024 = $fx1024:expr,
        xany = $fxany:expr,
    ) => {
        paste::paste! {
            fn [< benchmark_ $name >](c: &mut Criterion) {
                c.bench_function(&format!("{} x1024", $name), |b| {
                    let (x, y) = utils::get_sample_vectors(1024);
                    b.iter(|| repeat!(1000, $fx1024, &x, &y));
                });
                c.bench_function(&format!("{} xany-1301", $name), |b| {
                    let (x, y) = utils::get_sample_vectors(1301);
                    b.iter(|| repeat!(1000, $fxany, &x, &y));
                });
                c.bench_function(&format!("{} xany-1024", $name), |b| {
                    let (x, y) = utils::get_sample_vectors(1024);
                    b.iter(|| repeat!(1000, $fxany, &x, &y));
                });
            }
        }
    };
}

benchmark_distance_measure!(
    "f32_avx2_nofma_dot",
    x1024 = f32_xconst_avx2_nofma_dot::<1024>,
    xany = f32_xany_avx2_nofma_dot,
);
benchmark_distance_measure!(
    "f32_avx2_fma_dot",
    x1024 = f32_xconst_avx2_fma_dot::<1024>,
    xany = f32_xany_avx2_fma_dot,
);
benchmark_distance_measure!(
    "f32_avx2_nofma_cosine",
    x1024 = f32_xconst_avx2_nofma_cosine::<1024>,
    xany = f32_xany_avx2_nofma_cosine,
);
benchmark_distance_measure!(
    "f32_avx2_fma_cosine",
    x1024 = f32_xconst_avx2_fma_cosine::<1024>,
    xany = f32_xany_avx2_fma_cosine,
);
benchmark_distance_measure!(
    "f32_avx2_nofma_euclidean",
    x1024 = f32_xconst_avx2_nofma_squared_euclidean::<1024>,
    xany = f32_xany_avx2_nofma_squared_euclidean,
);
benchmark_distance_measure!(
    "f32_avx2_fma_euclidean",
    x1024 = f32_xconst_avx2_fma_squared_euclidean::<1024>,
    xany = f32_xany_avx2_fma_squared_euclidean,
);

benchmark_distance_measure!(
    "f32_fallback_nofma_dot",
    x1024 = f32_xany_fallback_nofma_dot,
    xany = f32_xany_fallback_nofma_dot,
);
benchmark_distance_measure!(
    "f32_fallback_nofma_cosine",
    x1024 = f32_xany_fallback_nofma_cosine,
    xany = f32_xany_fallback_nofma_cosine,
);
benchmark_distance_measure!(
    "f32_fallback_nofma_euclidean",
    x1024 = f32_xany_fallback_nofma_squared_euclidean,
    xany = f32_xany_fallback_nofma_squared_euclidean,
);

fn benchmark_f32_xany_ndarray(c: &mut Criterion) {
    c.bench_function("ndarray x1024 dot", |b| {
        let (x, y) = utils::get_sample_vectors(1024);
        let x = ndarray::Array1::from_shape_vec((x.len(),), x).unwrap();
        let y = ndarray::Array1::from_shape_vec((y.len(),), y).unwrap();

        b.iter(|| repeat!(1000, ndarray::Array1::dot, &x, &y));
    });
    c.bench_function("ndarray x1024 cosine", |b| {
        let (x, y) = utils::get_sample_vectors(1024);
        let x = ndarray::Array1::from_shape_vec((x.len(),), x).unwrap();
        let y = ndarray::Array1::from_shape_vec((y.len(),), y).unwrap();

        b.iter(|| repeat!(1000, ndarray_cosine, &x, &y));
    });
    c.bench_function("ndarray x1024 euclidean", |b| {
        let (x, y) = utils::get_sample_vectors(1024);
        let x = ndarray::Array1::from_shape_vec((x.len(),), x).unwrap();
        let y = ndarray::Array1::from_shape_vec((y.len(),), y).unwrap();

        b.iter(|| repeat!(1000, ndarray_euclidean, &x, &y));
    });
}

fn benchmark_f32_xany_simsimd(c: &mut Criterion) {
    c.bench_function("simsimd x1024 dot", |b| {
        let (x, y) = utils::get_sample_vectors(1024);
        b.iter(|| repeat!(1000, simsimd_dot, &x, &y));
    });
    c.bench_function("simsimd x1024 cosine", |b| {
        let (x, y) = utils::get_sample_vectors(1024);
        b.iter(|| repeat!(1000, simsimd_cosine, &x, &y));
    });
    c.bench_function("simsimd x1024 euclidean", |b| {
        let (x, y) = utils::get_sample_vectors(1024);
        b.iter(|| repeat!(1000, simsimd_euclidean, &x, &y));
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10));
    targets =
        benchmark_f32_xany_ndarray,
        benchmark_f32_xany_simsimd,
        benchmark_f32_avx2_nofma_dot,
        benchmark_f32_avx2_fma_dot,
        benchmark_f32_avx2_nofma_cosine,
        benchmark_f32_avx2_fma_cosine,
        benchmark_f32_avx2_nofma_euclidean,
        benchmark_f32_avx2_fma_euclidean,
        benchmark_f32_fallback_nofma_dot,
        benchmark_f32_fallback_nofma_cosine,
        benchmark_f32_fallback_nofma_euclidean,
);
criterion_main!(benches);

fn ndarray_cosine(a: &ndarray::Array1<f32>, b: &ndarray::Array1<f32>) -> f32 {
    let norm_a = a.dot(a);
    let norm_b = b.dot(b);
    let dot = a.dot(b);
    utils::cosine::<_, AutoMath>(dot, norm_a, norm_b)
}

fn ndarray_euclidean(a: &ndarray::Array1<f32>, b: &ndarray::Array1<f32>) -> f32 {
    let diff = a.sub(b);
    diff.dot(&diff)
}

fn simsimd_dot(a: &[f32], b: &[f32]) -> f32 {
    simsimd::SpatialSimilarity::dot(a, b).unwrap_or_default() as _
}

fn simsimd_cosine(a: &[f32], b: &[f32]) -> f32 {
    simsimd::SpatialSimilarity::cosine(a, b).unwrap_or_default() as _
}

fn simsimd_euclidean(a: &[f32], b: &[f32]) -> f32 {
    simsimd::SpatialSimilarity::sqeuclidean(a, b).unwrap_or_default() as _
}
