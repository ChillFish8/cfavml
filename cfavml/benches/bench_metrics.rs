use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use eonn_accel::danger::*;

mod utils;

macro_rules! benchmark_metric {
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
            }
        }
    };
}

benchmark_metric!(
    "f32_avx2_nofma_min",
    x1024 = f32_xconst_avx2_nofma_min_horizontal::<1024>,
    xany = f32_xany_avx2_nofma_min_horizontal,
);
benchmark_metric!(
    "f32_avx2_nofma_max",
    x1024 = f32_xconst_avx2_nofma_max_horizontal::<1024>,
    xany = f32_xany_avx2_nofma_max_horizontal,
);
benchmark_metric!(
    "f32_avx2_nofma_sum",
    x1024 = f32_xconst_avx2_nofma_sum_horizontal::<1024>,
    xany = f32_xany_avx2_nofma_sum_horizontal,
);

benchmark_metric!(
    "f32_avx512_nofma_min",
    x1024 = f32_xconst_avx512_nofma_min_horizontal::<1024>,
    xany = f32_xany_avx512_nofma_min_horizontal,
);
benchmark_metric!(
    "f32_avx512_nofma_max",
    x1024 = f32_xconst_avx512_nofma_max_horizontal::<1024>,
    xany = f32_xany_avx512_nofma_max_horizontal,
);
benchmark_metric!(
    "f32_avx512_nofma_sum",
    x1024 = f32_xconst_avx512_nofma_sum_horizontal::<1024>,
    xany = f32_xany_avx512_nofma_sum_horizontal,
);

benchmark_metric!(
    "f32_fallback_nofma_min",
    x1024 = f32_xany_fallback_nofma_min_horizontal,
    xany = f32_xany_fallback_nofma_min_horizontal,
);
benchmark_metric!(
    "f32_fallback_nofma_max",
    x1024 = f32_xany_fallback_nofma_max_horizontal,
    xany = f32_xany_fallback_nofma_max_horizontal,
);
benchmark_metric!(
    "f32_fallback_nofma_sum",
    x1024 = f32_xany_fallback_nofma_sum_horizontal,
    xany = f32_xany_fallback_nofma_sum_horizontal,
);

criterion_group!(
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(60));
    targets =
        benchmark_f32_avx2_nofma_sum,
        benchmark_f32_avx2_nofma_max,
        benchmark_f32_avx2_nofma_min,
        benchmark_f32_avx512_nofma_sum,
        benchmark_f32_avx512_nofma_max,
        benchmark_f32_avx512_nofma_min,
        benchmark_f32_fallback_nofma_sum,
        benchmark_f32_fallback_nofma_max,
        benchmark_f32_fallback_nofma_min,
);
criterion_main!(benches);
