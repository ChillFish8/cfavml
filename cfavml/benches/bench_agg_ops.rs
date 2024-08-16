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
mod sum {
    use cfavml::safe_trait_agg_ops::AggOps;
    use ndarray::{Data, ViewRepr};
    use rand::distributions::{Distribution, Standard};

    use super::*;

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn ndarray<T>(bencher: Bencher)
    where
        T: Copy + num_traits::identities::Zero,
        Standard: Distribution<T>,
        for<'a> ViewRepr<&'a mut T>: Data<Elem = T>,
    {
        let (l1, _) = utils::get_sample_vectors::<T>(DIMS);
        let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();

        bencher.bench_local(|| {
            let l1_view = black_box(l1_view);

            l1_view.sum()
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn cfavml<T>(bencher: Bencher)
    where
        Standard: Distribution<T>,
        T: AggOps,
    {
        let (l1, _) = utils::get_sample_vectors::<T>(DIMS);

        bencher.bench_local(|| {
            let l1_view = black_box(l1.as_ref());

            cfavml::sum(l1_view)
        });
    }
}
