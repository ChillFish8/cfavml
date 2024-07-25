use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const SEED: u64 = 2837564324875;

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

pub fn basic_transpose<T: Copy + Default>(
    width: usize,
    height: usize,
    data: &[T],
) -> Vec<T> {
    let mut result = vec![T::default(); width * height];

    for i in 0..width {
        for j in 0..height {
            result[i * height + j] = data[j * width + i];
        }
    }

    result
}
