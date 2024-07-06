use rand::distributions::{Distribution, Standard};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::danger::cosine;
use crate::math::{AutoMath, Math};

const SEED: u64 = 34535345353;

pub fn get_sample_vectors<T>(size: usize) -> (Vec<T>, Vec<T>)
where
    Standard: Distribution<T>,
{
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);

    let mut x = Vec::new();
    let mut y = Vec::new();
    for _ in 0..size {
        x.push(rng.gen());
        y.push(rng.gen());
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
