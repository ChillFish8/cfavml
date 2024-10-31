use std::hint::black_box;

use divan::Bencher;
use divan::counter::ItemsCount;
use ndarray::Array3;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    divan::main();
}


#[divan::bench(
    sample_count = 32, 
    sample_size = 25,
    counters = [ItemsCount::new(2usize * 10 * 256 * 256)],
)]
fn bench_cfavml_two_array_no_broadcast_no_time_alloc(bencher: Bencher) {
    let a = Array3::random((10, 256, 256), Uniform::new(1.0, 10.0));
    let b = Array3::random((10, 256, 256), Uniform::new(1.0, 10.0));
    
    bencher
        .with_inputs(|| {
            a.clone()
        })
        .bench_local_values(|a| {
             cfavml_ndarray::ops::mul(a, black_box(&b))
        });
}

#[divan::bench(
    sample_count = 32, 
    sample_size = 25,
    counters = [ItemsCount::new(2usize * 10 * 256 * 256)],
)]
fn bench_cfavml_two_array_no_broadcast_time_alloc(bencher: Bencher) {
    let a = Array3::random((10, 256, 256), Uniform::new(1.0, 10.0));
    let b = Array3::random((10, 256, 256), Uniform::new(1.0, 10.0));

    bencher.bench_local(|| {
        let a = black_box(&a).clone();
        cfavml_ndarray::ops::mul(a, black_box(&b))
    });
}

#[divan::bench(
    sample_count = 32, 
    sample_size = 25,
    counters = [ItemsCount::new(2usize * 10 * 256 * 256)],
)]
fn bench_cfavml_one_array_broadcast_value_no_time_alloc(bencher: Bencher) {
    let a = Array3::random((10, 256, 256), Uniform::new(1.0, 10.0));

    bencher
        .with_inputs(|| {
            a.clone()
        })
        .bench_local_values(|a| {
            cfavml_ndarray::ops::mul(a, black_box(3.0))
        });
}

#[divan::bench(
    sample_count = 32, 
    sample_size = 25,
    counters = [ItemsCount::new(2usize * 10 * 256 * 256)],
)]
fn bench_cfavml_one_array_broadcast_value_time_alloc(bencher: Bencher) {
    let a = Array3::random((10, 256, 256), Uniform::new(1.0, 10.0));

    bencher.bench_local(|| {
        let a = black_box(&a).clone();
        cfavml_ndarray::ops::mul(a, black_box(3.0))
    });
}

#[divan::bench(
    sample_count = 32, 
    sample_size = 25,
    counters = [ItemsCount::new(2usize * 10 * 256 * 256)],
)]
fn bench_default_two_array_no_broadcast_no_time_alloc(bencher: Bencher) {
    let a = Array3::random((10, 256, 256), Uniform::new(1.0, 10.0));
    let b = Array3::random((10, 256, 256), Uniform::new(1.0, 10.0));

    bencher
        .with_inputs(|| {
            a.clone()
        })
        .bench_local_values(|a| {
            a * black_box(&b)
        });
}

#[divan::bench(
    sample_count = 32, 
    sample_size = 25,
    counters = [ItemsCount::new(2usize * 10 * 256 * 256)],
)]
fn bench_default_two_array_no_broadcast_time_alloc(bencher: Bencher) {
    let a = Array3::random((10, 256, 256), Uniform::new(1.0, 10.0));
    let b = Array3::random((10, 256, 256), Uniform::new(1.0, 10.0));

    bencher.bench_local(|| {
        let a = black_box(&a).clone();
        a * black_box(&b)
    });
}

#[divan::bench(
    sample_count = 32, 
    sample_size = 25,
    counters = [ItemsCount::new(2usize * 10 * 256 * 256)],
)]
fn bench_default_two_array_no_broadcast_two_views(bencher: Bencher) {
    let a = Array3::random((10, 256, 256), Uniform::new(1.0, 10.0));
    let b = Array3::random((10, 256, 256), Uniform::new(1.0, 10.0));

    bencher.bench_local(|| {
        let a = black_box(&a);
        let b = black_box(&b);
        a * b
    });
}

#[divan::bench(
    sample_count = 32, 
    sample_size = 25,
    counters = [ItemsCount::new(2usize * 10 * 256 * 256)],
)]
fn bench_default_one_array_broadcast_value_no_time_alloc(bencher: Bencher) {
    let a = Array3::random((10, 256, 256), Uniform::new(1.0, 10.0));

    bencher
        .with_inputs(|| {
            a.clone()
        })
        .bench_local_values(|a| {
            a * black_box(3.0)
        });
}

#[divan::bench(
    sample_count = 32, 
    sample_size = 25,
    counters = [ItemsCount::new(2usize * 10 * 256 * 256)],
)]
fn bench_default_one_array_broadcast_value_time_alloc(bencher: Bencher) {
    let a = Array3::random((10, 256, 256), Uniform::new(1.0, 10.0));

    bencher.bench_local(|| {
        let a = black_box(&a).clone();
        a * black_box(3.0)
    });
}

#[divan::bench(
    sample_count = 32, 
    sample_size = 25,
    counters = [ItemsCount::new(2usize * 10 * 256 * 256)],
)]
fn bench_default_one_array_broadcast_value_with_view(bencher: Bencher) {
    let a = Array3::random((10, 256, 256), Uniform::new(1.0, 10.0));

    bencher.bench_local(|| {
        let a = black_box(&a);
        let b = black_box(3.0);
        a * b
    });
}