#[cfg(unix)]
extern crate blas_src;

use std::hint::black_box;

use divan::counter::ItemsCount;
use divan::Bencher;
use ndarray::ArrayView1;

mod utils;

const DIMS: usize = 1536;
const CMP_VALUE: u16 = 124;

fn main() {
    divan::main();
}

#[divan::bench_group(
    sample_count = 500,
    sample_size = 5000,
    threads = false,
    counters = [ItemsCount::new(DIMS)],
)]
mod max {
    use cfavml::buffer::WriteOnlyBuffer;
    use cfavml::math::{Math, StdMath};
    use cfavml::mem_loader::IntoMemLoader;
    use cfavml::safe_trait_cmp_ops::CmpOps;
    use ndarray::{Array1, Data, ViewRepr};
    use rand::distributions::{Distribution, Standard};

    use super::*;

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn ndarray_horizontal<T>(bencher: Bencher)
    where
        T: Copy + num_traits::identities::Zero,
        StdMath: Math<T>,
        Standard: Distribution<T>,
        for<'a> ViewRepr<&'a mut T>: Data<Elem = T>,
    {
        let (l1, _) = utils::get_sample_vectors::<T>(DIMS);
        let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();

        bencher.bench_local(|| {
            let l1_view = black_box(l1_view);
            l1_view.fold(StdMath::min(), |a, b| StdMath::cmp_max(a, *b))
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn ndarray_value<T>(bencher: Bencher)
    where
        T: Copy + num_traits::identities::Zero + TryFrom<u16>,
        <T as TryFrom<u16>>::Error: core::fmt::Debug,
        StdMath: Math<T>,
        Standard: Distribution<T>,
        for<'a> ViewRepr<&'a mut T>: Data<Elem = T>,
    {
        let (l1, _) = utils::get_sample_vectors::<T>(DIMS);
        let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();
        let mut result = Array1::zeros((l1.len(),));

        bencher.bench_local(|| {
            let value = black_box(T::try_from(CMP_VALUE).unwrap());
            let l1_view = black_box(l1_view);
            let result = black_box(&mut result);

            ndarray::azip!((r in result, a in l1_view) *r = StdMath::cmp_max(*a, value));
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn ndarray_vector<T>(bencher: Bencher)
    where
        T: Copy + num_traits::identities::Zero,
        StdMath: Math<T>,
        Standard: Distribution<T>,
        for<'a> ViewRepr<&'a mut T>: Data<Elem = T>,
    {
        let (l1, l2) = utils::get_sample_vectors::<T>(DIMS);
        let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();
        let l2_view = ArrayView1::from_shape((l2.len(),), &l2).unwrap();
        let mut result = Array1::zeros((l2.len(),));

        bencher.bench_local(|| {
            let l1_view = black_box(l1_view);
            let l2_view = black_box(l2_view);
            let result = black_box(&mut result);

            ndarray::azip!((r in result, a in l1_view, b in l2_view) *r = StdMath::cmp_max(*a, *b));
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn cfavml_horizontal<T>(bencher: Bencher)
    where
        Standard: Distribution<T>,
        T: CmpOps,
    {
        let (l1, _) = utils::get_sample_vectors::<T>(DIMS);

        bencher.bench_local(|| cfavml::max(black_box(&l1)));
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn cfavml_value<T>(bencher: Bencher)
    where
        T: CmpOps + Default + Copy + TryFrom<u16> + IntoMemLoader<T>,
        Standard: Distribution<T>,
        <T as TryFrom<u16>>::Error: core::fmt::Debug,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        let (l1, _) = utils::get_sample_vectors::<T>(DIMS);
        let mut result = vec![T::default(); DIMS];

        bencher.bench_local(|| {
            let value = black_box(T::try_from(CMP_VALUE).unwrap());
            let result = black_box(&mut result);

            cfavml::max_vertical(value, black_box(&l1), result)
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn cfavml_vector<T>(bencher: Bencher)
    where
        Standard: Distribution<T>,
        T: CmpOps + Default + Copy,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        let (l1, l2) = utils::get_sample_vectors::<T>(DIMS);
        let mut result = vec![T::default(); DIMS];

        bencher.bench_local(|| {
            let result = black_box(&mut result);
            cfavml::max_vertical(black_box(&l1), black_box(&l2), result)
        });
    }
}

#[divan::bench_group(
    sample_count = 500,
    sample_size = 5000,
    threads = false,
    counters = [ItemsCount::new(DIMS)],
)]
mod min {
    use cfavml::buffer::WriteOnlyBuffer;
    use cfavml::math::{Math, StdMath};
    use cfavml::mem_loader::IntoMemLoader;
    use cfavml::safe_trait_cmp_ops::CmpOps;
    use ndarray::{Array1, Data, ViewRepr};
    use rand::distributions::{Distribution, Standard};

    use super::*;

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn ndarray_horizontal<T>(bencher: Bencher)
    where
        T: Copy + num_traits::identities::Zero,
        StdMath: Math<T>,
        Standard: Distribution<T>,
        for<'a> ViewRepr<&'a mut T>: Data<Elem = T>,
    {
        let (l1, _) = utils::get_sample_vectors::<T>(DIMS);
        let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();

        bencher.bench_local(|| {
            let l1_view = black_box(l1_view);
            l1_view.fold(StdMath::max(), |a, b| StdMath::cmp_min(a, *b))
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn ndarray_value<T>(bencher: Bencher)
    where
        T: Copy + num_traits::identities::Zero + TryFrom<u16>,
        <T as TryFrom<u16>>::Error: core::fmt::Debug,
        StdMath: Math<T>,
        Standard: Distribution<T>,
        for<'a> ViewRepr<&'a mut T>: Data<Elem = T>,
    {
        let (l1, _) = utils::get_sample_vectors::<T>(DIMS);
        let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();
        let mut result = Array1::zeros((l1.len(),));

        bencher.bench_local(|| {
            let value = black_box(T::try_from(CMP_VALUE).unwrap());
            let l1_view = black_box(l1_view);
            let result = black_box(&mut result);

            ndarray::azip!((r in result, a in l1_view) *r = StdMath::cmp_min(*a, value));
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn ndarray_vector<T>(bencher: Bencher)
    where
        T: Copy + num_traits::identities::Zero,
        StdMath: Math<T>,
        Standard: Distribution<T>,
        for<'a> ViewRepr<&'a mut T>: Data<Elem = T>,
    {
        let (l1, l2) = utils::get_sample_vectors::<T>(DIMS);
        let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();
        let l2_view = ArrayView1::from_shape((l2.len(),), &l2).unwrap();
        let mut result = Array1::zeros((l2.len(),));

        bencher.bench_local(|| {
            let l1_view = black_box(l1_view);
            let l2_view = black_box(l2_view);
            let result = black_box(&mut result);

            ndarray::azip!((r in result, a in l1_view, b in l2_view) *r = StdMath::cmp_min(*a, *b));
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn cfavml_horizontal<T>(bencher: Bencher)
    where
        Standard: Distribution<T>,
        T: CmpOps,
    {
        let (l1, _) = utils::get_sample_vectors::<T>(DIMS);

        bencher.bench_local(|| cfavml::min(black_box(&l1)));
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn cfavml_value<T>(bencher: Bencher)
    where
        T: CmpOps + Default + Copy + TryFrom<u16> + IntoMemLoader<T>,
        Standard: Distribution<T>,
        <T as TryFrom<u16>>::Error: core::fmt::Debug,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        let (l1, _) = utils::get_sample_vectors::<T>(DIMS);
        let mut result = vec![T::default(); DIMS];

        bencher.bench_local(|| {
            let value = black_box(T::try_from(CMP_VALUE).unwrap());
            let result = black_box(&mut result);

            cfavml::min_vertical(value, black_box(&l1), result)
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn cfavml_vector<T>(bencher: Bencher)
    where
        Standard: Distribution<T>,
        T: CmpOps + Default + Copy,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        let (l1, l2) = utils::get_sample_vectors::<T>(DIMS);
        let mut result = vec![T::default(); DIMS];

        bencher.bench_local(|| {
            let result = black_box(&mut result);
            cfavml::min_vertical(black_box(&l1), black_box(&l2), result)
        });
    }
}

#[divan::bench_group(
    sample_count = 500,
    sample_size = 5000,
    threads = false,
    counters = [ItemsCount::new(DIMS)],
)]
mod eq {
    use cfavml::buffer::WriteOnlyBuffer;
    use cfavml::math::{Math, StdMath};
    use cfavml::mem_loader::IntoMemLoader;
    use cfavml::safe_trait_cmp_ops::CmpOps;
    use ndarray::{Array1, Data, ViewRepr};
    use rand::distributions::{Distribution, Standard};

    use super::*;

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn ndarray_value<T>(bencher: Bencher)
    where
        T: Copy + num_traits::identities::Zero + TryFrom<u16>,
        <T as TryFrom<u16>>::Error: core::fmt::Debug,
        StdMath: Math<T>,
        Standard: Distribution<T>,
        for<'a> ViewRepr<&'a mut T>: Data<Elem = T>,
    {
        let (l1, _) = utils::get_sample_vectors::<T>(DIMS);
        let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();
        let mut result = Array1::zeros((l1.len(),));

        bencher.bench_local(|| {
            let value = black_box(T::try_from(CMP_VALUE).unwrap());
            let l1_view = black_box(l1_view);
            let result = black_box(&mut result);

            ndarray::azip!((r in result, a in l1_view) *r = StdMath::cast_bool(StdMath::cmp_eq(*a, value)));
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn ndarray_vector<T>(bencher: Bencher)
    where
        T: Copy + num_traits::identities::Zero,
        StdMath: Math<T>,
        Standard: Distribution<T>,
        for<'a> ViewRepr<&'a mut T>: Data<Elem = T>,
    {
        let (l1, l2) = utils::get_sample_vectors::<T>(DIMS);
        let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();
        let l2_view = ArrayView1::from_shape((l2.len(),), &l2).unwrap();
        let mut result = Array1::zeros((l2.len(),));

        bencher.bench_local(|| {
            let l1_view = black_box(l1_view);
            let l2_view = black_box(l2_view);
            let result = black_box(&mut result);

            ndarray::azip!((r in result, a in l1_view, b in l2_view) *r = StdMath::cast_bool(StdMath::cmp_eq(*a, *b)));
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn cfavml_value<T>(bencher: Bencher)
    where
        T: CmpOps + Default + Copy + TryFrom<u16> + IntoMemLoader<T>,
        Standard: Distribution<T>,
        <T as TryFrom<u16>>::Error: core::fmt::Debug,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        let (l1, _) = utils::get_sample_vectors::<T>(DIMS);
        let mut result = vec![T::default(); DIMS];

        bencher.bench_local(|| {
            let value = black_box(T::try_from(CMP_VALUE).unwrap());
            let result = black_box(&mut result);

            cfavml::eq_vertical(value, black_box(&l1), result)
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn cfavml_vector<T>(bencher: Bencher)
    where
        Standard: Distribution<T>,
        T: CmpOps + Default + Copy,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        let (l1, l2) = utils::get_sample_vectors::<T>(DIMS);
        let mut result = vec![T::default(); DIMS];

        bencher.bench_local(|| {
            let result = black_box(&mut result);
            cfavml::eq_vertical(black_box(&l1), black_box(&l2), result)
        });
    }
}

#[divan::bench_group(
    sample_count = 500,
    sample_size = 5000,
    threads = false,
    counters = [ItemsCount::new(DIMS)],
)]
mod neq {
    use cfavml::buffer::WriteOnlyBuffer;
    use cfavml::math::{Math, StdMath};
    use cfavml::mem_loader::IntoMemLoader;
    use cfavml::safe_trait_cmp_ops::CmpOps;
    use ndarray::{Array1, Data, ViewRepr};
    use rand::distributions::{Distribution, Standard};

    use super::*;

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn ndarray_value<T>(bencher: Bencher)
    where
        T: Copy + num_traits::identities::Zero + TryFrom<u16>,
        <T as TryFrom<u16>>::Error: core::fmt::Debug,
        StdMath: Math<T>,
        Standard: Distribution<T>,
        for<'a> ViewRepr<&'a mut T>: Data<Elem = T>,
    {
        let (l1, _) = utils::get_sample_vectors::<T>(DIMS);
        let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();
        let mut result = Array1::zeros((l1.len(),));

        bencher.bench_local(|| {
            let value = black_box(T::try_from(CMP_VALUE).unwrap());
            let l1_view = black_box(l1_view);
            let result = black_box(&mut result);

            ndarray::azip!((r in result, a in l1_view) *r = StdMath::cast_bool(!StdMath::cmp_eq(*a, value)));
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn ndarray_vector<T>(bencher: Bencher)
    where
        T: Copy + num_traits::identities::Zero,
        StdMath: Math<T>,
        Standard: Distribution<T>,
        for<'a> ViewRepr<&'a mut T>: Data<Elem = T>,
    {
        let (l1, l2) = utils::get_sample_vectors::<T>(DIMS);
        let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();
        let l2_view = ArrayView1::from_shape((l2.len(),), &l2).unwrap();
        let mut result = Array1::zeros((l2.len(),));

        bencher.bench_local(|| {
            let l1_view = black_box(l1_view);
            let l2_view = black_box(l2_view);
            let result = black_box(&mut result);

            ndarray::azip!((r in result, a in l1_view, b in l2_view) *r = StdMath::cast_bool(!StdMath::cmp_eq(*a, *b)));
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn cfavml_value<T>(bencher: Bencher)
    where
        T: CmpOps + Default + Copy + TryFrom<u16> + IntoMemLoader<T>,
        Standard: Distribution<T>,
        <T as TryFrom<u16>>::Error: core::fmt::Debug,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        let (l1, _) = utils::get_sample_vectors::<T>(DIMS);
        let mut result = vec![T::default(); DIMS];

        bencher.bench_local(|| {
            let value = black_box(T::try_from(CMP_VALUE).unwrap());
            let result = black_box(&mut result);

            cfavml::neq_vertical(value, black_box(&l1), result)
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn cfavml_vector<T>(bencher: Bencher)
    where
        Standard: Distribution<T>,
        T: CmpOps + Default + Copy,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        let (l1, l2) = utils::get_sample_vectors::<T>(DIMS);
        let mut result = vec![T::default(); DIMS];

        bencher.bench_local(|| {
            let result = black_box(&mut result);
            cfavml::neq_vertical(black_box(&l1), black_box(&l2), result)
        });
    }
}

#[divan::bench_group(
    sample_count = 500,
    sample_size = 5000,
    threads = false,
    counters = [ItemsCount::new(DIMS)],
)]
mod lt {
    use cfavml::buffer::WriteOnlyBuffer;
    use cfavml::math::{Math, StdMath};
    use cfavml::mem_loader::IntoMemLoader;
    use cfavml::safe_trait_cmp_ops::CmpOps;
    use ndarray::{Array1, Data, ViewRepr};
    use rand::distributions::{Distribution, Standard};

    use super::*;

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn ndarray_value<T>(bencher: Bencher)
    where
        T: Copy + num_traits::identities::Zero + TryFrom<u16>,
        <T as TryFrom<u16>>::Error: core::fmt::Debug,
        StdMath: Math<T>,
        Standard: Distribution<T>,
        for<'a> ViewRepr<&'a mut T>: Data<Elem = T>,
    {
        let (l1, _) = utils::get_sample_vectors::<T>(DIMS);
        let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();
        let mut result = Array1::zeros((l1.len(),));

        bencher.bench_local(|| {
            let value = black_box(T::try_from(CMP_VALUE).unwrap());
            let l1_view = black_box(l1_view);
            let result = black_box(&mut result);

            ndarray::azip!((r in result, a in l1_view) *r = StdMath::cast_bool(StdMath::cmp_lt(*a, value)));
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn ndarray_vector<T>(bencher: Bencher)
    where
        T: Copy + num_traits::identities::Zero,
        StdMath: Math<T>,
        Standard: Distribution<T>,
        for<'a> ViewRepr<&'a mut T>: Data<Elem = T>,
    {
        let (l1, l2) = utils::get_sample_vectors::<T>(DIMS);
        let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();
        let l2_view = ArrayView1::from_shape((l2.len(),), &l2).unwrap();
        let mut result = Array1::zeros((l2.len(),));

        bencher.bench_local(|| {
            let l1_view = black_box(l1_view);
            let l2_view = black_box(l2_view);
            let result = black_box(&mut result);

            ndarray::azip!((r in result, a in l1_view, b in l2_view) *r = StdMath::cast_bool(StdMath::cmp_lt(*a, *b)));
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn cfavml_value<T>(bencher: Bencher)
    where
        T: CmpOps + Default + Copy + TryFrom<u16> + IntoMemLoader<T>,
        Standard: Distribution<T>,
        <T as TryFrom<u16>>::Error: core::fmt::Debug,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        let (l1, _) = utils::get_sample_vectors::<T>(DIMS);
        let mut result = vec![T::default(); DIMS];

        bencher.bench_local(|| {
            let value = black_box(T::try_from(CMP_VALUE).unwrap());
            let result = black_box(&mut result);

            cfavml::lt_vertical(value, black_box(&l1), result)
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn cfavml_vector<T>(bencher: Bencher)
    where
        Standard: Distribution<T>,
        T: CmpOps + Default + Copy,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        let (l1, l2) = utils::get_sample_vectors::<T>(DIMS);
        let mut result = vec![T::default(); DIMS];

        bencher.bench_local(|| {
            let result = black_box(&mut result);
            cfavml::lt_vertical(black_box(&l1), black_box(&l2), result)
        });
    }
}

#[divan::bench_group(
    sample_count = 500,
    sample_size = 5000,
    threads = false,
    counters = [ItemsCount::new(DIMS)],
)]
mod lte {
    use cfavml::buffer::WriteOnlyBuffer;
    use cfavml::math::{Math, StdMath};
    use cfavml::mem_loader::IntoMemLoader;
    use cfavml::safe_trait_cmp_ops::CmpOps;
    use ndarray::{Array1, Data, ViewRepr};
    use rand::distributions::{Distribution, Standard};

    use super::*;

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn ndarray_value<T>(bencher: Bencher)
    where
        T: Copy + num_traits::identities::Zero + TryFrom<u16>,
        <T as TryFrom<u16>>::Error: core::fmt::Debug,
        StdMath: Math<T>,
        Standard: Distribution<T>,
        for<'a> ViewRepr<&'a mut T>: Data<Elem = T>,
    {
        let (l1, _) = utils::get_sample_vectors::<T>(DIMS);
        let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();
        let mut result = Array1::zeros((l1.len(),));

        bencher.bench_local(|| {
            let value = black_box(T::try_from(CMP_VALUE).unwrap());
            let l1_view = black_box(l1_view);
            let result = black_box(&mut result);

            ndarray::azip!((r in result, a in l1_view) *r = StdMath::cast_bool(StdMath::cmp_lte(*a, value)));
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn ndarray_vector<T>(bencher: Bencher)
    where
        T: Copy + num_traits::identities::Zero,
        StdMath: Math<T>,
        Standard: Distribution<T>,
        for<'a> ViewRepr<&'a mut T>: Data<Elem = T>,
    {
        let (l1, l2) = utils::get_sample_vectors::<T>(DIMS);
        let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();
        let l2_view = ArrayView1::from_shape((l2.len(),), &l2).unwrap();
        let mut result = Array1::zeros((l2.len(),));

        bencher.bench_local(|| {
            let l1_view = black_box(l1_view);
            let l2_view = black_box(l2_view);
            let result = black_box(&mut result);

            ndarray::azip!((r in result, a in l1_view, b in l2_view) *r = StdMath::cast_bool(StdMath::cmp_lte(*a, *b)));
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn cfavml_value<T>(bencher: Bencher)
    where
        T: CmpOps + Default + Copy + TryFrom<u16> + IntoMemLoader<T>,
        Standard: Distribution<T>,
        <T as TryFrom<u16>>::Error: core::fmt::Debug,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        let (l1, _) = utils::get_sample_vectors::<T>(DIMS);
        let mut result = vec![T::default(); DIMS];

        bencher.bench_local(|| {
            let value = black_box(T::try_from(CMP_VALUE).unwrap());
            let result = black_box(&mut result);

            cfavml::lte_vertical(value, black_box(&l1), result)
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn cfavml_vector<T>(bencher: Bencher)
    where
        Standard: Distribution<T>,
        T: CmpOps + Default + Copy,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        let (l1, l2) = utils::get_sample_vectors::<T>(DIMS);
        let mut result = vec![T::default(); DIMS];

        bencher.bench_local(|| {
            let result = black_box(&mut result);
            cfavml::lte_vertical(black_box(&l1), black_box(&l2), result)
        });
    }
}

#[divan::bench_group(
    sample_count = 500,
    sample_size = 5000,
    threads = false,
    counters = [ItemsCount::new(DIMS)],
)]
mod gt {
    use cfavml::buffer::WriteOnlyBuffer;
    use cfavml::math::{Math, StdMath};
    use cfavml::mem_loader::IntoMemLoader;
    use cfavml::safe_trait_cmp_ops::CmpOps;
    use ndarray::{Array1, Data, ViewRepr};
    use rand::distributions::{Distribution, Standard};

    use super::*;

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn ndarray_value<T>(bencher: Bencher)
    where
        T: Copy + num_traits::identities::Zero + TryFrom<u16>,
        <T as TryFrom<u16>>::Error: core::fmt::Debug,
        StdMath: Math<T>,
        Standard: Distribution<T>,
        for<'a> ViewRepr<&'a mut T>: Data<Elem = T>,
    {
        let (l1, _) = utils::get_sample_vectors::<T>(DIMS);
        let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();
        let mut result = Array1::zeros((l1.len(),));

        bencher.bench_local(|| {
            let value = black_box(T::try_from(CMP_VALUE).unwrap());
            let l1_view = black_box(l1_view);
            let result = black_box(&mut result);

            ndarray::azip!((r in result, a in l1_view) *r = StdMath::cast_bool(StdMath::cmp_gt(*a, value)));
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn ndarray_vector<T>(bencher: Bencher)
    where
        T: Copy + num_traits::identities::Zero,
        StdMath: Math<T>,
        Standard: Distribution<T>,
        for<'a> ViewRepr<&'a mut T>: Data<Elem = T>,
    {
        let (l1, l2) = utils::get_sample_vectors::<T>(DIMS);
        let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();
        let l2_view = ArrayView1::from_shape((l2.len(),), &l2).unwrap();
        let mut result = Array1::zeros((l2.len(),));

        bencher.bench_local(|| {
            let l1_view = black_box(l1_view);
            let l2_view = black_box(l2_view);
            let result = black_box(&mut result);

            ndarray::azip!((r in result, a in l1_view, b in l2_view) *r = StdMath::cast_bool(StdMath::cmp_gt(*a, *b)));
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn cfavml_value<T>(bencher: Bencher)
    where
        T: CmpOps + Default + Copy + TryFrom<u16> + IntoMemLoader<T>,
        Standard: Distribution<T>,
        <T as TryFrom<u16>>::Error: core::fmt::Debug,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        let (l1, _) = utils::get_sample_vectors::<T>(DIMS);
        let mut result = vec![T::default(); DIMS];

        bencher.bench_local(|| {
            let value = black_box(T::try_from(CMP_VALUE).unwrap());
            let result = black_box(&mut result);

            cfavml::gt_vertical(value, black_box(&l1), result)
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn cfavml_vector<T>(bencher: Bencher)
    where
        Standard: Distribution<T>,
        T: CmpOps + Default + Copy,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        let (l1, l2) = utils::get_sample_vectors::<T>(DIMS);
        let mut result = vec![T::default(); DIMS];

        bencher.bench_local(|| {
            let result = black_box(&mut result);
            cfavml::gt_vertical(black_box(&l1), black_box(&l2), result)
        });
    }
}

#[divan::bench_group(
    sample_count = 500,
    sample_size = 5000,
    threads = false,
    counters = [ItemsCount::new(DIMS)],
)]
mod gte {
    use cfavml::buffer::WriteOnlyBuffer;
    use cfavml::math::{Math, StdMath};
    use cfavml::mem_loader::IntoMemLoader;
    use cfavml::safe_trait_cmp_ops::CmpOps;
    use ndarray::{Array1, Data, ViewRepr};
    use rand::distributions::{Distribution, Standard};

    use super::*;

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn ndarray_value<T>(bencher: Bencher)
    where
        T: Copy + num_traits::identities::Zero + TryFrom<u16>,
        <T as TryFrom<u16>>::Error: core::fmt::Debug,
        StdMath: Math<T>,
        Standard: Distribution<T>,
        for<'a> ViewRepr<&'a mut T>: Data<Elem = T>,
    {
        let (l1, _) = utils::get_sample_vectors::<T>(DIMS);
        let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();
        let mut result = Array1::zeros((l1.len(),));

        bencher.bench_local(|| {
            let value = black_box(T::try_from(CMP_VALUE).unwrap());
            let l1_view = black_box(l1_view);
            let result = black_box(&mut result);

            ndarray::azip!((r in result, a in l1_view) *r = StdMath::cast_bool(StdMath::cmp_gte(*a, value)));
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn ndarray_vector<T>(bencher: Bencher)
    where
        T: Copy + num_traits::identities::Zero,
        StdMath: Math<T>,
        Standard: Distribution<T>,
        for<'a> ViewRepr<&'a mut T>: Data<Elem = T>,
    {
        let (l1, l2) = utils::get_sample_vectors::<T>(DIMS);
        let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();
        let l2_view = ArrayView1::from_shape((l2.len(),), &l2).unwrap();
        let mut result = Array1::zeros((l2.len(),));

        bencher.bench_local(|| {
            let l1_view = black_box(l1_view);
            let l2_view = black_box(l2_view);
            let result = black_box(&mut result);

            ndarray::azip!((r in result, a in l1_view, b in l2_view) *r = StdMath::cast_bool(StdMath::cmp_gte(*a, *b)));
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn cfavml_value<T>(bencher: Bencher)
    where
        T: CmpOps + Default + Copy + TryFrom<u16> + IntoMemLoader<T>,
        Standard: Distribution<T>,
        <T as TryFrom<u16>>::Error: core::fmt::Debug,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        let (l1, _) = utils::get_sample_vectors::<T>(DIMS);
        let mut result = vec![T::default(); DIMS];

        bencher.bench_local(|| {
            let value = black_box(T::try_from(CMP_VALUE).unwrap());
            let result = black_box(&mut result);

            cfavml::gte_vertical(value, black_box(&l1), result)
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn cfavml_vector<T>(bencher: Bencher)
    where
        Standard: Distribution<T>,
        T: CmpOps + Default + Copy,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        let (l1, l2) = utils::get_sample_vectors::<T>(DIMS);
        let mut result = vec![T::default(); DIMS];

        bencher.bench_local(|| {
            let result = black_box(&mut result);
            cfavml::gte_vertical(black_box(&l1), black_box(&l2), result)
        });
    }
}
