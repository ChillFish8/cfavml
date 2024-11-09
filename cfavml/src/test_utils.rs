use core::ops::Neg;

use num_traits::{zero, Float, FloatConst};
use rand::distributions::uniform::SampleUniform;
use rand::distributions::{Distribution, Standard};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::danger::cosine;
use crate::math::{AutoMath, Math};
const SEED: u64 = 34535345353;

pub fn get_sample_vectors<T>(size: usize) -> (Vec<T>, Vec<T>)
where
    T: Copy,
    AutoMath: Math<T>,
    Standard: Distribution<T>,
{
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);

    let mut x = Vec::new();
    let mut y = Vec::new();
    for _ in 0..size {
        let mut v1 = rng.gen();
        let mut v2 = rng.gen();

        if AutoMath::cmp_eq(v1, AutoMath::zero()) {
            v1 = AutoMath::one();
        }

        if AutoMath::cmp_eq(v2, AutoMath::zero()) {
            v2 = AutoMath::one();
        }

        x.push(v1);
        y.push(v2);
    }

    (x, y)
}

pub fn get_subnormal_sample_vectors<T>(size: usize) -> (Vec<T>, Vec<T>)
where
    T: Copy + Float + FloatConst + Neg<Output = T> + SampleUniform,
    AutoMath: Math<T>,
    Standard: Distribution<T>,
{
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);

    let mut x = Vec::new();
    let mut y = Vec::new();
    for _ in 0..size {
        let mut v1 = rng.gen_range(core::ops::Range {
            start: zero(),
            end: T::min_positive_value(),
        });
        let mut v2 = rng.gen();

        if AutoMath::cmp_eq(v1, AutoMath::zero()) {
            v1 = AutoMath::one();
        }

        if AutoMath::cmp_eq(v2, AutoMath::zero()) {
            v2 = AutoMath::one();
        }

        x.push(v1);
        y.push(v2);
    }

    (x, y)
}

pub fn simple_dot<T>(x: &[T], y: &[T]) -> T
where
    T: Copy,
    AutoMath: Math<T>,
{
    let mut dot_product = AutoMath::zero();

    for i in 0..x.len() {
        dot_product = AutoMath::add(dot_product, AutoMath::mul(x[i], y[i]));
    }

    dot_product
}

pub fn simple_cosine<T>(x: &[T], y: &[T]) -> T
where
    T: Copy,
    AutoMath: Math<T>,
{
    let mut dot_product = AutoMath::zero();
    let mut norm_x = AutoMath::zero();
    let mut norm_y = AutoMath::zero();

    for i in 0..x.len() {
        dot_product = AutoMath::add(dot_product, AutoMath::mul(x[i], y[i]));
        norm_x = AutoMath::add(norm_x, AutoMath::mul(x[i], x[i]));
        norm_y = AutoMath::add(norm_y, AutoMath::mul(y[i], y[i]));
    }

    cosine::<_, AutoMath>(dot_product, norm_x, norm_y)
}

pub fn simple_euclidean<T>(x: &[T], y: &[T]) -> T
where
    T: Copy,
    AutoMath: Math<T>,
{
    let mut dist = AutoMath::zero();

    for i in 0..x.len() {
        let diff = AutoMath::sub(x[i], y[i]);
        dist = AutoMath::add(dist, AutoMath::mul(diff, diff));
    }

    dist
}
