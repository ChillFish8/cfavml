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
mod add {
    use std::ops::Add;

    use cfavml::buffer::WriteOnlyBuffer;
    use cfavml::math::{Math, StdMath};
    use cfavml::safe_trait_arithmetic_ops::ArithmeticOps;
    use ndarray::{Array1, Data, DataMut, ViewRepr};
    use rand::distributions::{Distribution, Standard};

    use super::*;

    const VALUE: u16 = 124;

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn ndarray_value<T>(bencher: Bencher)
    where
        T: Copy
            + num_traits::identities::Zero
            + TryFrom<u16>
            + ndarray::ScalarOperand
            + std::ops::AddAssign,
        <T as TryFrom<u16>>::Error: core::fmt::Debug,
        StdMath: Math<T>,
        Standard: Distribution<T>,
        for<'a> ViewRepr<&'a mut T>: Data<Elem = T> + DataMut,
    {
        let (l1, _) = utils::get_sample_vectors::<T>(DIMS);
        let l1_view = Array1::from_shape_vec((l1.len(),), l1).unwrap();

        bencher
            .with_inputs(|| l1_view.clone())
            .bench_local_values(|mut l1_view| {
                let value = black_box(T::try_from(VALUE).unwrap());
                l1_view += value;
            });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn ndarray_vector<T>(bencher: Bencher)
    where
        T: Copy + num_traits::identities::Zero + Add<Output = T>,
        StdMath: Math<T>,
        Standard: Distribution<T>,
    {
        let (l1, l2) = utils::get_sample_vectors::<T>(DIMS);
        let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();
        let l2_view = ArrayView1::from_shape((l2.len(),), &l2).unwrap();
        let mut result = Array1::zeros((l2.len(),));

        bencher.bench_local(|| {
            let l1_view = black_box(l1_view);
            let l2_view = black_box(l2_view);
            let result = black_box(&mut result);

            ndarray::azip!((r in result, a in l1_view, b in l2_view) *r = *a + *b);
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn cfavml_value<T>(bencher: Bencher)
    where
        T: ArithmeticOps + Default + Copy + TryFrom<u16>,
        Standard: Distribution<T>,
        <T as TryFrom<u16>>::Error: core::fmt::Debug,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        let (l1, _) = utils::get_sample_vectors::<T>(DIMS);
        let mut result = vec![T::default(); DIMS];

        bencher.bench_local(|| {
            let value = black_box(T::try_from(VALUE).unwrap());
            let l1_view = black_box(l1.as_ref());
            let result = black_box(&mut result);

            cfavml::add_value(value, l1_view, result)
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn cfavml_vector<T>(bencher: Bencher)
    where
        Standard: Distribution<T>,
        T: ArithmeticOps + Default + Copy,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        let (l1, l2) = utils::get_sample_vectors::<T>(DIMS);
        let mut result = vec![T::default(); DIMS];

        bencher.bench_local(|| {
            let l1_view = black_box(l1.as_ref());
            let l2_view = black_box(l2.as_ref());
            let result = black_box(&mut result);
            cfavml::add_vector(l1_view, l2_view, result)
        });
    }
}

#[divan::bench_group(
    sample_count = 500,
    sample_size = 5000,
    threads = false,
    counters = [ItemsCount::new(DIMS)],
)]
mod sub {
    use std::ops::Sub;

    use cfavml::buffer::WriteOnlyBuffer;
    use cfavml::math::{Math, StdMath};
    use cfavml::safe_trait_arithmetic_ops::ArithmeticOps;
    use ndarray::{Array1, Data, DataMut, ViewRepr};
    use rand::distributions::{Distribution, Standard};

    use super::*;

    const VALUE: u16 = 124;

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn ndarray_value<T>(bencher: Bencher)
    where
        T: Copy
            + num_traits::identities::Zero
            + TryFrom<u16>
            + ndarray::ScalarOperand
            + std::ops::SubAssign,
        <T as TryFrom<u16>>::Error: core::fmt::Debug,
        StdMath: Math<T>,
        Standard: Distribution<T>,
        for<'a> ViewRepr<&'a mut T>: Data<Elem = T> + DataMut,
    {
        let (l1, _) = utils::get_sample_vectors::<T>(DIMS);
        let l1_view = Array1::from_shape_vec((l1.len(),), l1).unwrap();

        bencher
            .with_inputs(|| l1_view.clone())
            .bench_local_values(|mut l1_view| {
                let value = black_box(T::try_from(VALUE).unwrap());
                l1_view -= value;
            });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn ndarray_vector<T>(bencher: Bencher)
    where
        T: Copy + num_traits::identities::Zero + Sub<Output = T>,
        StdMath: Math<T>,
        Standard: Distribution<T>,
    {
        let (l1, l2) = utils::get_sample_vectors::<T>(DIMS);
        let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();
        let l2_view = ArrayView1::from_shape((l2.len(),), &l2).unwrap();
        let mut result = Array1::zeros((l2.len(),));

        bencher.bench_local(|| {
            let l1_view = black_box(l1_view);
            let l2_view = black_box(l2_view);
            let result = black_box(&mut result);

            ndarray::azip!((r in result, a in l1_view, b in l2_view) *r = *a - *b);
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn cfavml_value<T>(bencher: Bencher)
    where
        T: ArithmeticOps + Default + Copy + TryFrom<u16>,
        Standard: Distribution<T>,
        <T as TryFrom<u16>>::Error: core::fmt::Debug,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        let (l1, _) = utils::get_sample_vectors::<T>(DIMS);
        let mut result = vec![T::default(); DIMS];

        bencher.bench_local(|| {
            let value = black_box(T::try_from(VALUE).unwrap());
            let l1_view = black_box(l1.as_ref());
            let result = black_box(&mut result);

            cfavml::add_value(value, l1_view, result)
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn cfavml_vector<T>(bencher: Bencher)
    where
        Standard: Distribution<T>,
        T: ArithmeticOps + Default + Copy,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        let (l1, l2) = utils::get_sample_vectors::<T>(DIMS);
        let mut result = vec![T::default(); DIMS];

        bencher.bench_local(|| {
            let l1_view = black_box(l1.as_ref());
            let l2_view = black_box(l2.as_ref());
            let result = black_box(&mut result);
            cfavml::add_vector(l1_view, l2_view, result)
        });
    }
}

#[divan::bench_group(
    sample_count = 500,
    sample_size = 5000,
    threads = false,
    counters = [ItemsCount::new(DIMS)],
)]
mod mul {
    use std::ops::Mul;

    use cfavml::buffer::WriteOnlyBuffer;
    use cfavml::math::{Math, StdMath};
    use cfavml::safe_trait_arithmetic_ops::ArithmeticOps;
    use ndarray::{Array1, Data, DataMut, ViewRepr};
    use rand::distributions::{Distribution, Standard};

    use super::*;

    const VALUE: u16 = 124;

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn ndarray_value<T>(bencher: Bencher)
    where
        T: Copy
            + num_traits::identities::Zero
            + TryFrom<u16>
            + ndarray::ScalarOperand
            + std::ops::MulAssign,
        <T as TryFrom<u16>>::Error: core::fmt::Debug,
        StdMath: Math<T>,
        Standard: Distribution<T>,
        for<'a> ViewRepr<&'a mut T>: Data<Elem = T> + DataMut,
    {
        let (l1, _) = utils::get_sample_vectors::<T>(DIMS);
        let l1_view = Array1::from_shape_vec((l1.len(),), l1).unwrap();

        bencher
            .with_inputs(|| l1_view.clone())
            .bench_local_values(|mut l1_view| {
                let value = black_box(T::try_from(VALUE).unwrap());
                l1_view *= value;
            });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn ndarray_vector<T>(bencher: Bencher)
    where
        T: Copy + num_traits::identities::Zero + Mul<Output = T>,
        StdMath: Math<T>,
        Standard: Distribution<T>,
    {
        let (l1, l2) = utils::get_sample_vectors::<T>(DIMS);
        let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();
        let l2_view = ArrayView1::from_shape((l2.len(),), &l2).unwrap();
        let mut result = Array1::zeros((l2.len(),));

        bencher.bench_local(|| {
            let l1_view = black_box(l1_view);
            let l2_view = black_box(l2_view);
            let result = black_box(&mut result);

            ndarray::azip!((r in result, a in l1_view, b in l2_view) *r = *a * *b);
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn cfavml_value<T>(bencher: Bencher)
    where
        T: ArithmeticOps + Default + Copy + TryFrom<u16>,
        Standard: Distribution<T>,
        <T as TryFrom<u16>>::Error: core::fmt::Debug,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        let (l1, _) = utils::get_sample_vectors::<T>(DIMS);
        let mut result = vec![T::default(); DIMS];

        bencher.bench_local(|| {
            let value = black_box(T::try_from(VALUE).unwrap());
            let l1_view = black_box(l1.as_ref());
            let result = black_box(&mut result);

            cfavml::mul_value(value, l1_view, result)
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn cfavml_vector<T>(bencher: Bencher)
    where
        Standard: Distribution<T>,
        T: ArithmeticOps + Default + Copy,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        let (l1, l2) = utils::get_sample_vectors::<T>(DIMS);
        let mut result = vec![T::default(); DIMS];

        bencher.bench_local(|| {
            let l1_view = black_box(l1.as_ref());
            let l2_view = black_box(l2.as_ref());
            let result = black_box(&mut result);
            cfavml::mul_vector(l1_view, l2_view, result)
        });
    }
}

#[divan::bench_group(
    sample_count = 500,
    sample_size = 5000,
    threads = false,
    counters = [ItemsCount::new(DIMS)],
)]
mod div {
    use std::ops::Div;

    use cfavml::buffer::WriteOnlyBuffer;
    use cfavml::math::{Math, StdMath};
    use cfavml::safe_trait_arithmetic_ops::ArithmeticOps;
    use ndarray::{Array1, Data, DataMut, ViewRepr};
    use rand::distributions::{Distribution, Standard};

    use super::*;

    const VALUE: u16 = 124;

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn ndarray_value<T>(bencher: Bencher)
    where
        T: Copy
            + num_traits::identities::Zero
            + TryFrom<u16>
            + ndarray::ScalarOperand
            + std::ops::DivAssign,
        <T as TryFrom<u16>>::Error: core::fmt::Debug,
        StdMath: Math<T>,
        Standard: Distribution<T>,
        for<'a> ViewRepr<&'a mut T>: Data<Elem = T> + DataMut,
    {
        let (l1, _) = utils::get_nonzero_sample_vectors::<T>(DIMS);
        let l1_view = Array1::from_shape_vec((l1.len(),), l1).unwrap();

        bencher
            .with_inputs(|| l1_view.clone())
            .bench_local_values(|mut l1_view| {
                let value = black_box(T::try_from(VALUE).unwrap());
                l1_view /= value;
            });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn ndarray_vector<T>(bencher: Bencher)
    where
        T: Copy + num_traits::identities::Zero + Div<Output = T>,
        StdMath: Math<T>,
        Standard: Distribution<T>,
    {
        let (l1, l2) = utils::get_nonzero_sample_vectors::<T>(DIMS);
        let l1_view = ArrayView1::from_shape((l1.len(),), &l1).unwrap();
        let l2_view = ArrayView1::from_shape((l2.len(),), &l2).unwrap();
        let mut result = Array1::zeros((l2.len(),));

        bencher.bench_local(|| {
            let l1_view = black_box(l1_view);
            let l2_view = black_box(l2_view);
            let result = black_box(&mut result);

            ndarray::azip!((r in result, a in l1_view, b in l2_view) *r = *a / *b);
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn cfavml_value<T>(bencher: Bencher)
    where
        T: ArithmeticOps + Default + Copy + TryFrom<u16>,
        StdMath: Math<T>,
        Standard: Distribution<T>,
        <T as TryFrom<u16>>::Error: core::fmt::Debug,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        let (l1, _) = utils::get_nonzero_sample_vectors::<T>(DIMS);
        let mut result = vec![T::default(); DIMS];

        bencher.bench_local(|| {
            let value = black_box(T::try_from(VALUE).unwrap());
            let l1_view = black_box(l1.as_ref());
            let result = black_box(&mut result);

            cfavml::div_value(value, l1_view, result)
        });
    }

    #[divan::bench(types = [f32, f64, i8, i16, i32, i64, u8, u16, u32, u64])]
    fn cfavml_vector<T>(bencher: Bencher)
    where
        T: ArithmeticOps + Default + Copy,
        Standard: Distribution<T>,
        StdMath: Math<T>,
        for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
    {
        let (l1, l2) = utils::get_nonzero_sample_vectors::<T>(DIMS);
        let mut result = vec![T::default(); DIMS];

        bencher.bench_local(|| {
            let l1_view = black_box(l1.as_ref());
            let l2_view = black_box(l2.as_ref());
            let result = black_box(&mut result);
            cfavml::div_vector(l1_view, l2_view, result)
        });
    }
}
