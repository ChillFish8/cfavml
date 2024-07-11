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

#[cfg(not(feature = "benchmark-aligned"))]
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

#[cfg(feature = "benchmark-aligned")]
pub fn get_sample_vectors<T>(
    size: usize,
) -> (aligned::AlignedBuffer<T>, aligned::AlignedBuffer<T>)
where
    Standard: Distribution<T>,
{
    use std::ops::DerefMut;

    let mut rng = ChaCha8Rng::seed_from_u64(SEED);

    let mut x = aligned::AlignedBuffer::new(size);
    let mut y = aligned::AlignedBuffer::new(size);

    let x_buf = x.deref_mut();
    let y_buf = y.deref_mut();
    for i in 0..size {
        x_buf[i] = rng.gen();
        y_buf[i] = rng.gen();
    }

    (x, y)
}

use cfavml::math::Math;

#[cfg(feature = "benchmark-aligned")]
pub mod aligned {
    use std::fmt::Debug;
    use std::iter::repeat;
    use std::marker::PhantomData;
    use std::mem;
    use std::ops::{Deref, DerefMut};

    #[derive(Debug, Default, Copy, Clone)]
    #[repr(C, align(64))]
    struct AlignTo64([u64; 8]);

    pub struct AlignedBuffer<T> {
        size: usize,
        phantom: PhantomData<T>,
        inner: Vec<AlignTo64>,
    }

    impl<T> AlignedBuffer<T> {
        pub fn new(size: usize) -> Self {
            Self {
                size,
                phantom: PhantomData,
                inner: aligned_vec::<T>(size),
            }
        }
    }

    impl<T> AsRef<[T]> for AlignedBuffer<T> {
        fn as_ref(&self) -> &[T] {
            self.deref()
        }
    }

    impl<T> Deref for AlignedBuffer<T> {
        type Target = [T];

        fn deref(&self) -> &Self::Target {
            let ptr = self.inner.as_ptr();
            unsafe { std::slice::from_raw_parts::<T>(ptr.cast(), self.size) }
        }
    }

    impl<T> DerefMut for AlignedBuffer<T> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            let ptr = self.inner.as_mut_ptr();
            unsafe { std::slice::from_raw_parts_mut::<T>(ptr.cast(), self.size) }
        }
    }

    fn aligned_vec<T>(n_elements: usize) -> Vec<AlignTo64> {
        let chunk_size = 64 / mem::size_of::<T>();

        // Lazy math to ensure we always have enough.
        let n_units = (n_elements / chunk_size) + 1;

        let mut aligned: Vec<AlignTo64> = Vec::with_capacity(n_units);
        aligned.extend(repeat(AlignTo64::default()).take(n_units));

        aligned
    }
}

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
