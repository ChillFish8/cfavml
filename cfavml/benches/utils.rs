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

#[cfg(not(feature = "benchmark-aligned"))]
pub fn get_sample_vectors(size: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);

    let mut x = Vec::new();
    let mut y = Vec::new();
    for _ in 0..size {
        x.push(rng.gen());
        y.push(rng.gen());
    }

    (x, y)
}

#[cfg(feature = "benchmark-aligned")]
pub fn get_sample_vectors(size: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED);

    let mut x = unsafe { aligned_vec(size) };
    let mut y = unsafe { aligned_vec(size) };
    for _ in 0..size {
        x.push(rng.gen());
        y.push(rng.gen());
    }

    (x, y)
}

#[cfg(feature = "benchmark-aligned")]
use std::mem;

#[cfg(feature = "benchmark-aligned")]
#[repr(C, align(64))]
struct AlignToSixtyFour([f32; 16]);

#[cfg(feature = "benchmark-aligned")]
unsafe fn aligned_vec(n_elements: usize) -> Vec<f32> {
    // Lazy math to ensure we always have enough.
    let n_units = (n_elements / 16) + 1;

    let mut aligned: Vec<AlignToSixtyFour> = Vec::with_capacity(n_units);

    let ptr = aligned.as_mut_ptr();
    let len_units = aligned.len();
    let cap_units = aligned.capacity();

    mem::forget(aligned);

    Vec::from_raw_parts(ptr as *mut f32, len_units * 16, cap_units * 16)
}
