use cfavml::math::{Math, StdMath};
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const SEED: u64 = 2837564324875;

#[macro_export]
macro_rules! repeat {
    ($n:expr, $func:expr, $($x:expr $(,)?)*) => {{
        #[allow(unused_unsafe)]
        unsafe {
            for _ in 0..$n {
                black_box($func($(black_box($x),)*));
            }
        }
    }};
}

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

#[allow(unused)]
pub fn get_nonzero_sample_vectors<T>(size: usize) -> (Vec<T>, Vec<T>)
where
    Standard: Distribution<T>,
    StdMath: Math<T>,
    T: Copy,
{
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);

    let mut x = Vec::new();
    let mut y = Vec::new();
    for _ in 0..size {
        let mut v1 = rng.gen();
        let mut v2 = rng.gen();
        while StdMath::cmp_eq(StdMath::zero(), v1)
            || StdMath::cmp_eq(StdMath::zero(), v2)
        {
            v1 = rng.gen();
            v2 = rng.gen();
        }

        x.push(v1);
        y.push(v2);
    }

    (x, y)
}

#[allow(unused)]
#[inline(always)]
pub fn cosine<T: Copy, M: Math<T>>(dot_product: T, norm_x: T, norm_y: T) -> T {
    if M::cmp_eq(norm_x, M::zero()) && M::cmp_eq(norm_y, M::zero()) {
        M::zero()
    } else if M::cmp_eq(norm_x, M::zero()) || M::cmp_eq(norm_y, M::zero()) {
        M::zero()
    } else {
        M::sub(
            M::one(),
            M::div(dot_product, M::sqrt(M::mul(norm_x, norm_y))),
        )
    }
}
