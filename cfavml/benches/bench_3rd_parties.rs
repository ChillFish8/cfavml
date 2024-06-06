#[cfg(unix)]
extern crate blas_src;

use std::hint::black_box;
use std::ops::Sub;

use criterion::{criterion_group, criterion_main, Criterion};
use simsimd::SpatialSimilarity;

mod utils;

fn benchmark_3rd_party_impls(c: &mut Criterion) {
    c.bench_function("cosine ndarray x1024 auto", |b| {
        let (x, y) = utils::get_sample_vectors(1024);
        let x = ndarray::Array1::from_shape_vec((1024,), x).unwrap();
        let y = ndarray::Array1::from_shape_vec((1024,), y).unwrap();
        b.iter(|| repeat!(1000, ndarray_cosine, &x, &y));
    });
    c.bench_function("cosine simsimd x1024 auto", |b| {
        let (x, y) = utils::get_sample_vectors(1024);
        b.iter(|| repeat!(1000, simsimd_cosine, &x, &y));
    });

    c.bench_function("dot ndarray x1024 auto", |b| {
        let (x, y) = utils::get_sample_vectors(1024);
        let x = ndarray::Array1::from_shape_vec((1024,), x).unwrap();
        let y = ndarray::Array1::from_shape_vec((1024,), y).unwrap();
        b.iter(|| repeat!(1000, ndarray_dot, &x, &y));
    });
    c.bench_function("dot simsimd x1024 auto", |b| {
        let (x, y) = utils::get_sample_vectors(1024);
        b.iter(|| repeat!(1000, simsimd_dot, &x, &y));
    });

    c.bench_function("euclidean ndarray x1024 auto", |b| {
        let (x, y) = utils::get_sample_vectors(1024);
        let x = ndarray::Array1::from_shape_vec((1024,), x).unwrap();
        let y = ndarray::Array1::from_shape_vec((1024,), y).unwrap();
        b.iter(|| repeat!(1000, ndarray_euclidean, &x, &y));
    });
    c.bench_function("euclidean simsimd x1024 auto", |b| {
        let (x, y) = utils::get_sample_vectors(1024);
        b.iter(|| repeat!(1000, simsimd_euclidean, &x, &y));
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = benchmark_3rd_party_impls
);
criterion_main!(benches);

fn ndarray_cosine(a: &ndarray::Array1<f32>, b: &ndarray::Array1<f32>) -> f32 {
    let dot_product = a.dot(b);
    let norm_x = a.dot(a);
    let norm_y = b.dot(b);

    if norm_x == 0.0 && norm_y == 0.0 {
        0.0
    } else if norm_x == 0.0 || norm_y == 0.0 {
        1.0
    } else {
        1.0 - (dot_product / (norm_x * norm_y).sqrt())
    }
}

fn simsimd_cosine(a: &[f32], b: &[f32]) -> f32 {
    f32::cosine(a, b).unwrap_or_default() as f32
}

fn ndarray_dot(a: &ndarray::Array1<f32>, b: &ndarray::Array1<f32>) -> f32 {
    a.dot(b)
}

fn simsimd_dot(a: &[f32], b: &[f32]) -> f32 {
    f32::dot(a, b).unwrap_or_default() as f32
}

fn ndarray_euclidean(a: &ndarray::Array1<f32>, b: &ndarray::Array1<f32>) -> f32 {
    let diff = a.sub(b);
    diff.dot(&diff)
}

fn simsimd_euclidean(a: &[f32], b: &[f32]) -> f32 {
    f32::sqeuclidean(a, b).unwrap_or_default() as f32
}
