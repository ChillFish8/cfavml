use std::hint::black_box;

use divan::Bencher;
use ndarray::ArrayView1;

mod utils;

const DIMS: usize = 4342;

fn main() {
    divan::main();
}

#[cfg_attr(
    not(debug_assertions),
    divan::bench_group(
        sample_count = 500,
        sample_size = 10000,
        threads = false,
        counters = [divan::counter::ItemsCount::new(DIMS)],
    )
)]
mod op_min {
    use super::*;

    macro_rules! ndarray_impls {
        ($t:ty) => {
            paste::paste! {
                #[cfg_attr(not(debug_assertions), divan::bench)]
                fn [< $t _min_ndarray>](bencher: Bencher) {
                    let (l1, _) = utils::get_sample_vectors::<$t>(DIMS);
                    let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();

                    bencher.bench_local(|| {
                        let l1_view = black_box(l1_view);
                        l1_view.fold(<cfavml::math::AutoMath as cfavml::math::Math<$t>>::max(), |a, b| a.min(*b))
                    });
                }
            }
        };
    }

    ndarray_impls!(f32);
    ndarray_impls!(f64);
    ndarray_impls!(u8);
    ndarray_impls!(u16);
    ndarray_impls!(u32);
    ndarray_impls!(u64);
    ndarray_impls!(i8);
    ndarray_impls!(i16);
    ndarray_impls!(i32);
    ndarray_impls!(i64);

    macro_rules! cfavml_impls {
        ($t:ty) => {
            paste::paste! {
                #[cfg_attr(not(debug_assertions), divan::bench)]
                fn [< $t _min_cfavml>](bencher: Bencher) {
                    let (l1, _) = utils::get_sample_vectors::<$t>(DIMS);

                    bencher.bench_local(|| {
                        let l1_view = black_box(&l1);
                        cfavml::[< $t _xany_min_horizontal >](l1_view)
                    });
                }
            }
        };
    }
    
    cfavml_impls!(f32);
    cfavml_impls!(f64);
    cfavml_impls!(u8);
    cfavml_impls!(u16);
    cfavml_impls!(u32);
    cfavml_impls!(u64);
    cfavml_impls!(i8);
    cfavml_impls!(i16);
    cfavml_impls!(i32);
    cfavml_impls!(i64);
}

#[cfg_attr(
    not(debug_assertions),
    divan::bench_group(
        sample_count = 500,
        sample_size = 10000,
        threads = false,
        counters = [divan::counter::ItemsCount::new(DIMS)]
    )
)]
mod op_max {
    use super::*;

    macro_rules! ndarray_impls {
        ($t:ty) => {
            paste::paste! {
                #[cfg_attr(not(debug_assertions), divan::bench)]
                fn [< $t _max_ndarray>](bencher: Bencher) {
                    let (l1, _) = utils::get_sample_vectors::<$t>(DIMS);
                    let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();

                    bencher.bench_local(|| {
                        let l1_view = black_box(l1_view);
                        l1_view.fold(<cfavml::math::AutoMath as cfavml::math::Math<$t>>::min(), |a, b| a.max(*b))
                    });
                }
            }
        };
    }

    ndarray_impls!(f32);
    ndarray_impls!(f64);
    ndarray_impls!(u8);
    ndarray_impls!(u16);
    ndarray_impls!(u32);
    ndarray_impls!(u64);
    ndarray_impls!(i8);
    ndarray_impls!(i16);
    ndarray_impls!(i32);
    ndarray_impls!(i64);

    macro_rules! cfavml_impls {
        ($t:ty) => {
            paste::paste! {
                #[cfg_attr(not(debug_assertions), divan::bench)]
                fn [< $t _max_cfavml>](bencher: Bencher) {
                    let (l1, _) = utils::get_sample_vectors::<$t>(DIMS);

                    bencher.bench_local(|| {
                        let l1_view = black_box(&l1);
                        cfavml::[< $t _xany_max_horizontal >](l1_view)
                    });
                }
            }
        };
    }

    cfavml_impls!(f32);
    cfavml_impls!(f64);
    cfavml_impls!(u8);
    cfavml_impls!(u16);
    cfavml_impls!(u32);
    cfavml_impls!(u64);
    cfavml_impls!(i8);
    cfavml_impls!(i16);
    cfavml_impls!(i32);
    cfavml_impls!(i64);
}

#[cfg_attr(
    not(debug_assertions),
    divan::bench_group(
        sample_count = 500,
        sample_size = 10000,
        threads = false,
        counters = [divan::counter::ItemsCount::new(DIMS)]
    )
)]
mod op_sum {
    use super::*;

    macro_rules! naive_impls {
        ($t:ty) => {
            paste::paste! {
                #[cfg_attr(not(debug_assertions), divan::bench)]
                fn [< $t _sum_naive>](bencher: Bencher) {
                    let (l1, _) = utils::get_sample_vectors::<$t>(DIMS);

                    bencher.bench_local(|| {
                        let l1_view = black_box(&l1);
                        l1_view.iter().sum::<$t>()
                    });
                }
            }
        };
    }

    naive_impls!(f32);
    naive_impls!(f64);
    naive_impls!(u8);
    naive_impls!(u16);
    naive_impls!(u32);
    naive_impls!(u64);
    naive_impls!(i8);
    naive_impls!(i16);
    naive_impls!(i32);
    naive_impls!(i64);

    macro_rules! ndarray_impls {
        ($t:ty) => {
            paste::paste! {
                #[cfg_attr(not(debug_assertions), divan::bench)]
                fn [< $t _sum_ndarray>](bencher: Bencher) {
                    let (l1, _) = utils::get_sample_vectors::<$t>(DIMS);
                    let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();

                    bencher.bench_local(|| {
                        let l1_view = black_box(l1_view);
                        l1_view.sum()
                    });
                }
            }
        };
    }

    ndarray_impls!(f32);
    ndarray_impls!(f64);
    ndarray_impls!(u8);
    ndarray_impls!(u16);
    ndarray_impls!(u32);
    ndarray_impls!(u64);
    ndarray_impls!(i8);
    ndarray_impls!(i16);
    ndarray_impls!(i32);
    ndarray_impls!(i64);

    macro_rules! cfavml_impls {
        ($t:ty) => {
            paste::paste! {
                #[cfg_attr(not(debug_assertions), divan::bench)]
                fn [< $t _sum_cfavml>](bencher: Bencher) {
                    let (l1, _) = utils::get_sample_vectors::<$t>(DIMS);

                    bencher.bench_local(|| {
                        let l1_view = black_box(&l1);
                        cfavml::[< $t _xany_sum >](l1_view)
                    });
                }
            }
        };
    }

    cfavml_impls!(f32);
    cfavml_impls!(f64);
    cfavml_impls!(u8);
    cfavml_impls!(u16);
    cfavml_impls!(u32);
    cfavml_impls!(u64);
    cfavml_impls!(i8);
    cfavml_impls!(i16);
    cfavml_impls!(i32);
    cfavml_impls!(i64);
}
