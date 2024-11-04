use std::{fmt::Debug, iter::zip};

use crate::test_utils::get_sample_vectors;
use cfavml::{
    danger::SimdRegister,
    math::{AutoMath, Math},
};
use core::ops::{Add, Div, Mul, Neg, Rem, Sub};
use num_complex::Complex;
use num_traits::Float;
use rand::{distributions::Standard, prelude::Distribution};

pub trait ComplexMath<T>
where
    T: Float + Debug,
    AutoMath: Math<T>,
{
    fn to_scalars(value: Complex<T>) -> (T, T) {
        (value.re, value.im)
    }
    fn from_scalars(re: T, im: T) -> Complex<T> {
        Complex::new(re, im)
    }
    fn zero() -> Complex<T> {
        Complex::new(T::zero(), T::zero())
    }

    fn mul(a: Complex<T>, b: Complex<T>) -> Complex<T> {
        a.mul(b)
    }
    fn add(a: Complex<T>, b: Complex<T>) -> Complex<T> {
        a.add(b)
    }
    fn sub(a: Complex<T>, b: Complex<T>) -> Complex<T> {
        a.sub(b)
    }
    fn div(a: Complex<T>, b: Complex<T>) -> Complex<T> {
        a.div(b)
    }
    // I could reuse the `is_close` function from the `Math` trait, but it's
    // currently behind a `#[cfg(test)]` attribute
    fn is_close(a: Complex<T>, b: Complex<T>) -> bool;
}
pub struct CMath;
pub type CM = CMath;
impl ComplexMath<f32> for CMath {
    fn is_close(a: Complex<f32>, b: Complex<f32>) -> bool {
        fn close(a: f32, b: f32) -> bool {
            let max = a.max(b);
            let min = a.min(b);
            let diff = max - min;
            diff <= 0.00015
        }
        close(a.re, b.re) && close(a.im, b.im)
    }
}
impl ComplexMath<f64> for CMath {
    fn is_close(a: Complex<f64>, b: Complex<f64>) -> bool {
        fn close(a: f64, b: f64) -> bool {
            let max = a.max(b);
            let min = a.min(b);
            let diff = max - min;
            diff <= 0.00015
        }
        close(a.re, b.re) && close(a.im, b.im)
    }
}
pub(crate) unsafe fn test_suite_impl<T, R>()
where
    T: Float + Debug,
    R: SimdRegister<Complex<T>>,
    AutoMath: Math<T>,
    Standard: Distribution<T>,
    CM: ComplexMath<T>,
{
    let (small_sample_l1, small_sample_l2) =
        get_sample_vectors::<T>(R::elements_per_lane());
    let (large_sample_l1, large_sample_l2) =
        get_sample_vectors::<T>(R::elements_per_dense());

    // Single lane handling.
    {
        let l1 = R::load(small_sample_l1.as_ptr());
        let l2 = R::load(small_sample_l2.as_ptr());
        let res = R::add(l1, l2);
        let value = R::sum_to_value(res);
        let expected_value = zip(small_sample_l1.iter(), small_sample_l2.iter())
            .map(|(a, b)| CM::add(*a, *b))
            .fold(CM::zero(), |a, b| CM::add(a, b));
        assert!(
            CM::is_close(value, expected_value),
            "Addition and sum test failed on single task"
        );
    }

    {
        let l1 = R::load(small_sample_l1.as_ptr());
        let l2 = R::load(small_sample_l2.as_ptr());
        let res = R::sub(l1, l2);
        let value = R::sum_to_value(res);
        let expected_value = zip(small_sample_l1.iter(), small_sample_l2.iter())
            .map(|(a, b)| CM::sub(*a, *b))
            .fold(CM::zero(), |a, b| CM::add(a, b));
        assert!(
            CM::is_close(value, expected_value),
            "Subtraction and sum test failed on single task"
        );
    }

    {
        let l1 = R::load(small_sample_l1.as_ptr());
        let l2 = R::load(small_sample_l2.as_ptr());
        let res = R::mul(l1, l2);
        let value = R::sum_to_value(res);
        let expected_value = zip(small_sample_l1.iter(), small_sample_l2.iter())
            .map(|(a, b)| CM::mul(*a, *b))
            .fold(CM::zero(), |a, b| CM::add(a, b));
        assert!(
            CM::is_close(value, expected_value),
            "Multiplication and sum test failed on single task"
        );
    }

    {
        let l1 = R::load(small_sample_l1.as_ptr());
        let l2 = R::load(small_sample_l2.as_ptr());
        let l3 = R::filled(CM::zero());
        let res = R::fmadd(l1, l2, l3);
        let value = R::sum_to_value(res);
        let expected_value = zip(small_sample_l1.iter(), small_sample_l2.iter())
            .map(|(a, b)| CM::mul(*a, *b))
            .fold(CM::zero(), |a, b| CM::add(a, b));
        assert!(
            CM::is_close(value, expected_value),
            "Fmadd and sum test failed on single task"
        );
    }

    {
        let l1 = R::load(small_sample_l1.as_ptr());
        let l2 = R::load(small_sample_l2.as_ptr());
        let res = R::div(l1, l2);
        let value = R::sum_to_value(res);
        let expected_value = zip(small_sample_l1.iter(), small_sample_l2.iter())
            .map(|(a, b)| CM::div(*a, *b))
            .fold(CM::zero(), |a, b| CM::add(a, b));
        assert!(
            CM::is_close(value, expected_value),
            "Division and sum test failed on single task"
        );
    }

    {
        let l1 = R::load(small_sample_l1.as_ptr());
        let l2 = R::load(small_sample_l2.as_ptr());
        let res = R::add(l1, l2);

        let mut target_output = vec![CM::zero(); R::elements_per_lane()];
        R::write(target_output.as_mut_ptr(), res);

        let expected_output = zip(small_sample_l1.iter(), small_sample_l2.iter())
            .map(|(a, b)| CM::add(*a, *b))
            .collect::<Vec<_>>();

        assert_eq!(
            target_output, expected_output,
            "Single lane write failed task"
        );
    }

    // Dense lane handling.
    {
        let l1 = R::load_dense(large_sample_l1.as_ptr());
        let l2 = R::load_dense(large_sample_l2.as_ptr());
        let res = R::add_dense(l1, l2);
        let reg = R::sum_to_register(res);
        let value = R::sum_to_value(reg);

        let expected_value = zip(large_sample_l1.iter(), large_sample_l2.iter())
            .map(|(a, b)| CM::add(*a, *b))
            .fold(CM::zero(), |a, b| CM::add(a, b));

        assert!(
            CM::is_close(value, expected_value),
            "Addition and sum test failed on dense task"
        );
    }

    {
        let l1 = R::load_dense(large_sample_l1.as_ptr());
        let l2 = R::load_dense(large_sample_l2.as_ptr());
        let res = R::sub_dense(l1, l2);
        let reg = R::sum_to_register(res);
        let value = R::sum_to_value(reg);

        let expected_value = zip(large_sample_l1.iter(), large_sample_l2.iter())
            .map(|(a, b)| CM::sub(*a, *b))
            .fold(CM::zero(), |a, b| CM::add(a, b));

        assert!(
            CM::is_close(value, expected_value),
            "Subtraction and sum test failed on dense task"
        );
    }

    {
        let l1 = R::load_dense(large_sample_l1.as_ptr());
        let l2 = R::load_dense(large_sample_l2.as_ptr());
        let res = R::mul_dense(l1, l2);
        let reg = R::sum_to_register(res);
        let value = R::sum_to_value(reg);

        let expected_value = zip(large_sample_l1.iter(), large_sample_l2.iter())
            .map(|(a, b)| CM::mul(*a, *b))
            .fold(CM::zero(), |a, b| CM::add(a, b));

        assert!(
            CM::is_close(value, expected_value),
            "Multiplication and sum test failed on dense task"
        );
    }

    {
        let l1 = R::load_dense(large_sample_l1.as_ptr());
        let l2 = R::load_dense(large_sample_l2.as_ptr());
        let l3 = R::filled_dense(CM::zero());
        let res = R::fmadd_dense(l1, l2, l3);
        let reg = R::sum_to_register(res);
        let value = R::sum_to_value(reg);

        let expected_value = zip(large_sample_l1.iter(), large_sample_l2.iter())
            .map(|(a, b)| CM::mul(*a, *b))
            .fold(CM::zero(), |a, b| CM::add(a, b));

        assert!(
            CM::is_close(value, expected_value),
            "Fmadd and sum test failed on dense task"
        );
    }

    {
        let l1 = R::load_dense(large_sample_l1.as_ptr());
        let l2 = R::load_dense(large_sample_l2.as_ptr());
        let res = R::div_dense(l1, l2);
        let reg = R::sum_to_register(res);
        let value = R::sum_to_value(reg);

        let expected_value = zip(large_sample_l1.iter(), large_sample_l2.iter())
            .map(|(a, b)| CM::div(*a, *b))
            .fold(CM::zero(), |a, b| CM::add(a, b));

        assert!(
            CM::is_close(value, expected_value),
            "Division and sum test failed on dense task"
        );
    }

    {
        let l1 = R::load_dense(large_sample_l1.as_ptr());
        let l2 = R::load_dense(large_sample_l2.as_ptr());
        let res = R::add_dense(l1, l2);

        let mut target_output = vec![CM::zero(); R::elements_per_dense()];
        R::write_dense(target_output.as_mut_ptr(), res);

        let expected_output = zip(large_sample_l1.iter(), large_sample_l2.iter())
            .map(|(a, b)| CM::add(*a, *b))
            .collect::<Vec<_>>();

        assert_eq!(
            target_output, expected_output,
            "Dense lane write failed dense task"
        );
    }
}
