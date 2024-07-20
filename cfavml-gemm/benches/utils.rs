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