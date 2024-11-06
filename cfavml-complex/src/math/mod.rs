use core::ops::Neg;

use cfavml::math::{AutoMath, Math};
use num_complex::Complex;
pub type AutoComplex = DefaultComplexMath;
pub struct DefaultComplexMath;

pub trait ComplexMath<T>
where
    T: Copy + Neg<Output = T>,
    AutoMath: Math<T>,
{
    #[inline(always)]
    fn to_scalars(value: Complex<T>) -> (T, T) {
        (value.re, value.im)
    }
    #[inline(always)]
    fn from_scalars(re: T, im: T) -> Complex<T> {
        Complex::new(re, im)
    }
    fn zero() -> Complex<T> {
        Complex::new(AutoMath::zero(), AutoMath::zero())
    }

    fn mul(a: Complex<T>, b: Complex<T>) -> Complex<T> {
        Complex::new(
            AutoMath::sub(AutoMath::mul(a.re, b.re), AutoMath::mul(a.im, b.im)),
            AutoMath::add(AutoMath::mul(a.re, b.im), AutoMath::mul(a.im, b.re)),
        )
    }
    #[inline(always)]
    fn add(a: Complex<T>, b: Complex<T>) -> Complex<T> {
        Complex {
            re: AutoMath::add(a.re, b.re),
            im: AutoMath::add(a.im, b.im),
        }
    }

    #[inline(always)]
    fn sub(a: Complex<T>, b: Complex<T>) -> Complex<T> {
        Complex {
            re: AutoMath::sub(a.re, b.re),
            im: AutoMath::sub(a.im, b.im),
        }
    }
    #[inline(always)]
    fn norm(a: Complex<T>) -> T {
        AutoMath::sqrt(AutoMath::add(
            AutoMath::mul(a.re, a.re),
            AutoMath::mul(a.im, a.im),
        ))
    }
    #[inline(always)]
    fn inv(a: Complex<T>) -> Complex<T> {
        let sqrd_norm =
            AutoMath::add(AutoMath::mul(a.re, a.re), AutoMath::mul(a.im, a.im));
        let inv_re = AutoMath::div(a.re, sqrd_norm);
        let inv_im = AutoMath::div(-a.im, sqrd_norm);
        Complex::new(inv_re, inv_im)
    }
    //need to implement this separately so I don't require the std::ops::Neg trait
    fn div(a: Complex<T>, b: Complex<T>) -> Complex<T> {
        Self::mul(a, Self::inv(b))
    }

    #[cfg(test)]
    // I could reuse the `is_close` function from the `Math` trait, but it's
    // currently behind a `#[cfg(test)]` attribute
    fn is_close(a: Complex<T>, b: Complex<T>) -> bool;
}
#[cfg(test)]
impl ComplexMath<f32> for DefaultComplexMath {
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
#[cfg(test)]
impl ComplexMath<f64> for DefaultComplexMath {
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
