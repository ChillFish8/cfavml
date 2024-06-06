use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use eonn_accel::danger::*;

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
    x1024 = f32_xconst_avx2_nofma_euclidean::<1024>,
    xany = f32_xany_avx2_nofma_euclidean,
);
benchmark_distance_measure!(
    "f32_avx2_fma_euclidean",
    x1024 = f32_xconst_avx2_fma_euclidean::<1024>,
    xany = f32_xany_avx2_fma_euclidean,
);

benchmark_distance_measure!(
    "f32_avx512_fma_dot",
    x1024 = f32_xconst_avx512_fma_dot::<1024>,
    xany = f32_xany_avx512_fma_dot,
);
benchmark_distance_measure!(
    "f32_avx512_fma_cosine",
    x1024 = f32_xconst_avx512_fma_cosine::<1024>,
    xany = f32_xany_avx512_fma_cosine,
);
benchmark_distance_measure!(
    "f32_avx512_fma_euclidean",
    x1024 = f32_xconst_avx512_fma_euclidean::<1024>,
    xany = f32_xany_avx512_fma_euclidean,
);

benchmark_distance_measure!(
    "f32_fallback_nofma_dot",
    x1024 = generic_xany_fallback_nofma_dot,
    xany = generic_xany_fallback_nofma_dot,
);
benchmark_distance_measure!(
    "f32_fallback_nofma_cosine",
    x1024 = generic_xany_fallback_nofma_cosine,
    xany = generic_xany_fallback_nofma_cosine,
);
benchmark_distance_measure!(
    "f32_fallback_nofma_euclidean",
    x1024 = generic_xany_fallback_nofma_euclidean,
    xany = generic_xany_fallback_nofma_euclidean,
);

criterion_group!(
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(60));
    targets =
        benchmark_f32_avx2_nofma_dot,
        benchmark_f32_avx2_fma_dot,
        benchmark_f32_avx2_nofma_cosine,
        benchmark_f32_avx2_fma_cosine,
        benchmark_f32_avx2_nofma_euclidean,
        benchmark_f32_avx2_fma_euclidean,
        benchmark_f32_avx512_fma_dot,
        benchmark_f32_avx512_fma_cosine,
        benchmark_f32_avx512_fma_euclidean,
        benchmark_f32_fallback_nofma_dot,
        benchmark_f32_fallback_nofma_cosine,
        benchmark_f32_fallback_nofma_euclidean,
);
criterion_main!(benches);
