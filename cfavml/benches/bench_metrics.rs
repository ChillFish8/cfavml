use std::hint::black_box;
use std::time::Duration;

use cfavml::danger::*;
use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::Axis;

mod utils;


fn benchmark_ndarray(c: &mut Criterion) {
    c.bench_function("f32_ndarray_sum xany-1301", |b| {
        let (x, _) = utils::get_sample_vectors(1301);
        let x = ndarray::Array1::from_shape_vec((1301,), x).unwrap();
        b.iter(|| repeat!(1000, ndarray::ArrayBase::sum, &x));
    });
    c.bench_function("f32_ndarray_sum xany-x1024", |b| {
        let (x, _) = utils::get_sample_vectors(1024);
        let x = ndarray::Array1::from_shape_vec((1024,), x).unwrap();
        b.iter(|| repeat!(1000, ndarray::ArrayBase::sum, &x));
    });
    c.bench_function("f32_ndarray_sum xany-x1024 matrix", |b| {
        let (x, _) = utils::get_sample_vectors(1024 * 2048);
        let x = ndarray::Array2::from_shape_vec((2048, 1024), x).unwrap();
        b.iter(|| x.sum_axis(Axis(0)));
    });

    c.bench_function("f32_ndarray_min xany-1301", |b| {
        let (x, _) = utils::get_sample_vectors(1301);
        let x = ndarray::Array1::from_shape_vec((1301,), x).unwrap();
        b.iter(|| repeat!(1000, ndarray_fold_op, &x, f32::INFINITY, |v1, v2| v1.min(*v2)));
    });
    c.bench_function("f32_ndarray_min xany-x1024", |b| {
        let (x, _) = utils::get_sample_vectors(1024);
        let x = ndarray::Array1::from_shape_vec((1024,), x).unwrap();
        b.iter(|| repeat!(1000, ndarray_fold_op, &x, f32::INFINITY, |v1, v2| v1.min(*v2)));
    });

    c.bench_function("f32_ndarray_max xany-1301", |b| {
        let (x, _) = utils::get_sample_vectors(1301);
        let x = ndarray::Array1::from_shape_vec((1301,), x).unwrap();
        b.iter(|| repeat!(1000, ndarray_fold_op, &x, f32::INFINITY, |v1, v2| v1.max(*v2)));
    });
    c.bench_function("f32_ndarray_max xany-x1024", |b| {
        let (x, _) = utils::get_sample_vectors(1024);
        let x = ndarray::Array1::from_shape_vec((1024,), x).unwrap();
        b.iter(|| repeat!(1000, ndarray_fold_op, &x, f32::NEG_INFINITY, |v1, v2| v1.max(*v2)));
    });
}


#[inline(always)]
fn ndarray_fold_op(arr: &ndarray::Array1<f32>, init: f32, op: fn(f32, &f32) -> f32) -> f32 {
    arr.fold(init, op)
}


macro_rules! benchmark_horizontal_metric {
    (
        $name:expr,
        x1024 = $fx1024:expr,
        xany = $fxany:expr,
    ) => {
        paste::paste! {
            fn [< benchmark_ $name >](c: &mut Criterion) {
                c.bench_function(&format!("{} x1024", $name), |b| {
                    let (x, _) = utils::get_sample_vectors(1024);
                    b.iter(|| repeat!(1000, $fx1024, &x));
                });
                c.bench_function(&format!("{} xany-1301", $name), |b| {
                    let (x, _) = utils::get_sample_vectors(1301);
                    b.iter(|| repeat!(1000, $fxany, &x));
                });
                c.bench_function(&format!("{} xany-1024", $name), |b| {
                    let (x, _) = utils::get_sample_vectors(1024);
                    b.iter(|| repeat!(1000, $fxany, &x));
                });
                c.bench_function(&format!("{} x1024 matrix", $name), |b| {
                    let (x, _) = utils::get_sample_vectors(1024 * 2048);
                    b.iter(|| {
                        for i in (0..x.len()).step_by(1024) {
                            unsafe { black_box($fx1024(&x[i..i+1024])) };
                        }
                    });
                });
                c.bench_function(&format!("{} xany-1024 matrix", $name), |b| {
                    let (x, _) = utils::get_sample_vectors(1024 * 2048);
                    b.iter(|| {
                        for i in (0..x.len()).step_by(1024) {
                            unsafe { black_box($fxany(&x[i..i+1024])) };
                        }
                    });
                });
            }
        }
    };
}

benchmark_horizontal_metric!(
    "f32_avx2_nofma_min",
    x1024 = f32_xconst_avx2_nofma_min_horizontal::<1024>,
    xany = f32_xany_avx2_nofma_min_horizontal,
);
benchmark_horizontal_metric!(
    "f32_avx2_nofma_max",
    x1024 = f32_xconst_avx2_nofma_max_horizontal::<1024>,
    xany = f32_xany_avx2_nofma_max_horizontal,
);
benchmark_horizontal_metric!(
    "f32_avx2_nofma_sum",
    x1024 = f32_xconst_avx2_nofma_sum_horizontal::<1024>,
    xany = f32_xany_avx2_nofma_sum_horizontal,
);

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly", feature = "benchmark-avx512"))]
benchmark_horizontal_metric!(
    "f32_avx512_nofma_min",
    x1024 = f32_xconst_avx512_nofma_min_horizontal::<1024>,
    xany = f32_xany_avx512_nofma_min_horizontal,
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly", feature = "benchmark-avx512"))]
benchmark_horizontal_metric!(
    "f32_avx512_nofma_max",
    x1024 = f32_xconst_avx512_nofma_max_horizontal::<1024>,
    xany = f32_xany_avx512_nofma_max_horizontal,
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly", feature = "benchmark-avx512"))]
benchmark_horizontal_metric!(
    "f32_avx512_nofma_sum",
    x1024 = f32_xconst_avx512_nofma_sum_horizontal::<1024>,
    xany = f32_xany_avx512_nofma_sum_horizontal,
);

benchmark_horizontal_metric!(
    "f32_fallback_nofma_min",
    x1024 = generic_xany_fallback_nofma_min_horizontal,
    xany = generic_xany_fallback_nofma_min_horizontal,
);
benchmark_horizontal_metric!(
    "f32_fallback_nofma_max",
    x1024 = generic_xany_fallback_nofma_max_horizontal,
    xany = generic_xany_fallback_nofma_max_horizontal,
);
benchmark_horizontal_metric!(
    "f32_fallback_nofma_sum",
    x1024 = generic_xany_fallback_nofma_sum_horizontal,
    xany = generic_xany_fallback_nofma_sum_horizontal,
);

criterion_group!(
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(60));
    targets =
        benchmark_ndarray,
        benchmark_f32_fallback_nofma_sum,
        benchmark_f32_fallback_nofma_max,
        benchmark_f32_fallback_nofma_min,
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
criterion_group!(
    name = benches_avx2;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10));
    targets =
        benchmark_f32_avx2_nofma_sum,
        benchmark_f32_avx2_nofma_max,
        benchmark_f32_avx2_nofma_min,
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly", feature = "benchmark-avx512"))]
criterion_group!(
    name = benches_avx512;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10));
    targets =
        benchmark_f32_avx512_nofma_sum,
        benchmark_f32_avx512_nofma_max,
        benchmark_f32_avx512_nofma_min,
);
criterion_main!(
    benches,
    benches_avx2,
);
