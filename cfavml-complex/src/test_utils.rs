use cfavml::math::{AutoMath, Math};
use num_complex::Complex;
use num_traits::Float;
use rand::{
    distributions::{Distribution, Standard},
    Rng, SeedableRng,
};
use rand_chacha::ChaCha8Rng;

const SEED: u64 = 34535345353;
pub fn get_sample_vectors<T>(size: usize) -> (Vec<Complex<T>>, Vec<Complex<T>>)
where
    T: Float,
    AutoMath: Math<T>,
    Standard: Distribution<T>,
{
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);

    let mut x = Vec::new();
    let mut y = Vec::new();
    for _ in 0..size {
        let v1: Complex<T> = Complex {
            re: rng.gen::<T>(),
            im: rng.gen::<T>(),
        };
        let v2: Complex<T> = Complex {
            re: rng.gen::<T>(),
            im: rng.gen::<T>(),
        };

        for v in &mut [v1, v2] {
            if AutoMath::cmp_eq(v.re, AutoMath::zero()) {
                v.re = AutoMath::one();
            }
            if AutoMath::cmp_eq(v.im, AutoMath::zero()) {
                v.im = AutoMath::one();
            }
        }

        x.push(v1);
        y.push(v2);
    }

    (x, y)
}
