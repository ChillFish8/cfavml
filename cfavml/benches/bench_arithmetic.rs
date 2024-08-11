use std::hint::black_box;

use divan::Bencher;

mod utils;

const DIMS: usize = 1_000_000;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    divan::main();
}

#[cfg_attr(
    not(debug_assertions),
    divan::bench_group(
        sample_count = 50,
        sample_size = 10,
        threads = false,
        counters = [divan::counter::ItemsCount::new(DIMS)],
    )
)]
mod op_vector_x_value {
    use std::ops::{Add, Div, Mul, Sub};

    use super::*;

    macro_rules! cfavml_impls {
        ($t:ty, ops = $($op:ident $(,)?)+) => {
            $(
                paste::paste! {
                    #[cfg_attr(not(debug_assertions), divan::bench)]
                    fn [< $t _ $op _value_cfavml>](bencher: Bencher) {
                        let (l1, _) = utils::get_sample_vectors::<$t>(DIMS);

                        bencher.bench_local(|| {
                                let mut result = Vec::<$t>::with_capacity(black_box(DIMS));
                                unsafe { result.set_len(black_box(DIMS)) };

                                cfavml::[< $t _xany_ $op _value>](
                                    black_box(2 as $t),
                                    black_box(&l1),
                                    black_box(&mut result),
                                );

                                unsafe { std::mem::transmute::<_, Vec<$t>>(result) }
                            });
                    }
                }
            )*
        };
    }

    cfavml_impls!(f32, ops = add, sub, mul, div);
    cfavml_impls!(f64, ops = add, sub, mul, div);

    // We don't enable div here because cfavml currently just uses scalar ops
    // and division is just so flipping slow.
    cfavml_impls!(i8, ops = add, sub, mul);
    cfavml_impls!(i16, ops = add, sub, mul);
    cfavml_impls!(i32, ops = add, sub, mul);
    cfavml_impls!(i64, ops = add, sub, mul);
    cfavml_impls!(u8, ops = add, sub, mul);
    cfavml_impls!(u16, ops = add, sub, mul);
    cfavml_impls!(u32, ops = add, sub, mul);
    cfavml_impls!(u64, ops = add, sub, mul);

    macro_rules! std_impls {
        ($t:ty, ops = $($op:ident $(,)?)+) => {
            $(
                paste::paste! {
                    #[cfg_attr(not(debug_assertions), divan::bench)]
                    fn [< $t _ $op _value_std>](bencher: Bencher) {
                        let (l1, _) = utils::get_sample_vectors::<$t>(DIMS);
                        bencher.bench_local(|| {
                            black_box(&l1)
                                .iter()
                                .map(|v| v.$op(black_box(2 as $t)))
                                .collect::<Vec<_>>()
                        });
                    }
                }
            )*
        };
    }

    std_impls!(f32, ops = add, sub, mul, div);
    std_impls!(f64, ops = add, sub, mul, div);

    // We don't enable div here because cfavml currently just uses scalar ops
    // and division is just so flipping slow.
    std_impls!(i8, ops = add, sub, mul);
    std_impls!(i16, ops = add, sub, mul);
    std_impls!(i32, ops = add, sub, mul);
    std_impls!(i64, ops = add, sub, mul);
    std_impls!(u8, ops = add, sub, mul);
    std_impls!(u16, ops = add, sub, mul);
    std_impls!(u32, ops = add, sub, mul);
    std_impls!(u64, ops = add, sub, mul);
}

unsafe fn copy_to_buf<T>(buf: &mut [T], mut iter: impl Iterator<Item = T>) {
    let mut cursor = 0;
    while let Some(value) = iter.next() {
        unsafe {
            *buf.get_unchecked_mut(cursor) = value;
        }
        cursor += 1;
    }
}
