#[cfg(unix)]
extern crate blas_src;

use std::hint::black_box;

use divan::counter::ItemsCount;
use divan::Bencher;
use ndarray::ArrayView1;

mod utils;

const DIMS: usize = 1536;

fn main() {
    divan::main();
}

#[divan::bench_group(
    sample_count = 500,
    sample_size = 5000,
    threads = false,
    counters = [ItemsCount::new(DIMS)],
)]
mod dot_product {
    use cfavml::safe_trait_distance_ops::DistanceOps;
    use ndarray::linalg::Dot;
    use ndarray::{Data, ViewRepr};
    use rand::distributions::{Distribution, Standard};

    use super::*;

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn ndarray<T>(bencher: Bencher)
    where
        Standard: Distribution<T>,
        for<'a> ViewRepr<&'a mut T>: Data<Elem = T>,
        for<'a> ArrayView1<'a, T>: Dot<ArrayView1<'a, T>>,
    {
        let (l1, l2) = utils::get_sample_vectors::<T>(DIMS);
        let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();
        let l2_view = ArrayView1::from_shape((l2.len(),), &l2).unwrap();

        bencher.bench_local(|| {
            let l1_view = black_box(l1_view);
            let l2_view = black_box(l2_view);

            l1_view.dot(&l2_view)
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn cfavml<T>(bencher: Bencher)
    where
        Standard: Distribution<T>,
        T: DistanceOps,
    {
        let (l1, l2) = utils::get_sample_vectors::<T>(DIMS);

        bencher.bench_local(|| {
            let l1_view = black_box(l1.as_ref());
            let l2_view = black_box(l2.as_ref());

            cfavml::dot(l1_view, l2_view)
        });
    }
}

#[divan::bench_group(
    sample_count = 500,
    sample_size = 5000,
    threads = false,
    counters = [ItemsCount::new(DIMS)],
)]
mod cosine {
    use cfavml::math::*;
    use cfavml::safe_trait_distance_ops::DistanceOps;
    use ndarray::linalg::Dot;
    use ndarray::{Data, ViewRepr};
    use rand::distributions::{Distribution, Standard};

    use super::*;

    #[divan::bench(types = [f32, f64])]
    fn ndarray<T>(bencher: Bencher)
    where
        T: Copy + num_traits::identities::One,
        AutoMath: Math<T>,
        Standard: Distribution<T>,
        for<'a> ViewRepr<&'a mut T>: Data<Elem = T>,
        for<'a> ArrayView1<'a, T>: Dot<ArrayView1<'a, T>, Output = T>,
    {
        let (l1, l2) = utils::get_sample_vectors::<T>(DIMS);
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

    #[divan::bench(types = [f32, f64])]
    fn cfavml<T>(bencher: Bencher)
    where
        Standard: Distribution<T>,
        T: DistanceOps,
    {
        let (l1, l2) = utils::get_sample_vectors::<T>(DIMS);

        bencher.bench_local(|| {
            let l1_view = black_box(l1.as_ref());
            let l2_view = black_box(l2.as_ref());

            cfavml::cosine(l1_view, l2_view)
        });
    }
}

#[divan::bench_group(
    sample_count = 500,
    sample_size = 5000,
    threads = false,
    counters = [ItemsCount::new(DIMS)],
)]
mod euclidean {
    use std::ops::Sub;

    use cfavml::math::*;
    use cfavml::safe_trait_distance_ops::DistanceOps;
    use ndarray::linalg::Dot;
    use ndarray::{Array1, Data, OwnedRepr};
    use rand::distributions::{Distribution, Standard};

    use super::*;

    #[divan::bench(types = [f32, f64])]
    fn ndarray<T>(bencher: Bencher)
    where
        T: Copy + num_traits::identities::One,
        AutoMath: Math<T>,
        Standard: Distribution<T>,
        OwnedRepr<T>: Data<Elem = T>,
        Array1<T>: Dot<Array1<T>, Output = T>,
        for<'a> &'a Array1<T>: Sub<&'a Array1<T>, Output = Array1<T>>,
    {
        let (l1, l2) = utils::get_sample_vectors::<T>(DIMS);
        let l1_view = Array1::from_shape_vec((l1.len(),), l1.to_vec()).unwrap();
        let l2_view = Array1::from_shape_vec((l2.len(),), l2.to_vec()).unwrap();

        bencher.bench_local(|| {
            let l1_view = black_box(&l1_view);
            let l2_view = black_box(&l2_view);

            let diff = l1_view - l2_view;
            diff.dot(&diff)
        });
    }

    #[divan::bench(types = [f32, f64])]
    fn cfavml<T>(bencher: Bencher)
    where
        Standard: Distribution<T>,
        T: DistanceOps,
    {
        let (l1, l2) = utils::get_sample_vectors::<T>(DIMS);

        bencher.bench_local(|| {
            let l1_view = black_box(l1.as_ref());
            let l2_view = black_box(l2.as_ref());

            cfavml::squared_euclidean(l1_view, l2_view)
        });
    }
}
