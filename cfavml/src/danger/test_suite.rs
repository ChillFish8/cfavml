use std::fmt::Debug;

use super::*;
use crate::buffer::WriteOnlyBuffer;
use crate::danger::SimdRegister;
use crate::math::{AutoMath, Math};

const DATA_SIZE: usize = if cfg!(miri) { 133 } else { 1043 };

// Some types like `i8` and `i16` do not behave well with the rng generated vectors _and_
// cosine logic, so we skip them since it is unlikely anyone will actually do cosine distance
// on i8 or i16 values.
macro_rules! test_cosine_extra {
    ($t:ident, $im:ident) => {
        paste::paste! {
            #[test]
            fn [<test_ $im:lower _ $t _cosine>]() {
                let (l1, l2) = crate::test_utils::get_sample_vectors::<$t>(DATA_SIZE);
                unsafe { crate::danger::op_cosine::test_cosine::<$t, $im>(l1, l2) };
            }
        }
    };
}

// In cases like f32 and f64 where have have comparison we need to ensure that
// all implementations behave equivalently and consistently.
macro_rules! test_nan_sanity {
    ($t:ident, $im:ident) => {
        paste::paste! {
            #[test]
            fn [<test_ $im:lower _ $t _float_sanity>]() {
                let l1 = vec![1.0, 0.0, $t::NAN, $t::INFINITY, $t::NEG_INFINITY];
                let l2 = vec![1.0, 1.0, 1.0, 1.0, 1.0];

                test_cmp_value_all::<$t, $im>(l1.clone(), 0.0);
                test_cmp_vector_all::<$t, $im>(l1, l2);
            }
        }
    };
}

macro_rules! test_suite {
    ($t:ident, $im:ident) => {
        paste::paste! {
            #[test]
            fn [<test_ $im:lower _ $t _suite>]() {
                unsafe { crate::danger::impl_test::test_suite_impl::<$t, $im>() }
            }

            #[test]
            fn [<test_ $im:lower _ $t _dot>]() {
                let (l1, l2) = crate::test_utils::get_sample_vectors::<$t>(DATA_SIZE);
                unsafe { crate::danger::op_dot::test_dot::<$t, $im>(l1, l2) };
            }

            #[test]
            fn [<test_ $im:lower _ $t _norm>]() {
                let (l1, _) = crate::test_utils::get_sample_vectors::<$t>(DATA_SIZE);
                unsafe { crate::danger::op_norm::test_squared_norm::<$t, $im>(l1) };
            }

            #[test]
            fn [<test_ $im:lower _ $t _euclidean>]() {
                let (l1, l2) = crate::test_utils::get_sample_vectors::<$t>(DATA_SIZE);
                unsafe { crate::danger::op_euclidean::test_euclidean::<$t, $im>(l1, l2) };
            }

            #[test]
            fn [<test_ $im:lower _ $t _max>]() {
                let (l1, l2) = crate::test_utils::get_sample_vectors::<$t>(DATA_SIZE);
                unsafe { crate::danger::op_cmp_max::test_max::<$t, $im>(l1, l2) };
            }

            #[test]
            fn [<test_ $im:lower _ $t _min>]() {
                let (l1, l2) = crate::test_utils::get_sample_vectors::<$t>(DATA_SIZE);
                unsafe { crate::danger::op_cmp_min::test_min::<$t, $im>(l1, l2) };
            }

            #[test]
            fn [<test_ $im:lower _ $t _sum>]() {
                let l1 = vec![1 as $t; 1043];
                unsafe { crate::danger::op_sum::test_sum::<$t, $im>(l1) };
            }

            #[test]
            fn [<test_ $im:lower _ $t _arithmetic_value>]() {
                let (l1, _) = (vec![1 as $t; DATA_SIZE], vec![3 as $t; DATA_SIZE]);
                test_arithmetic_value_all::<$t, $im>(l1, 2 as $t);
            }

            #[test]
            fn [<test_ $im:lower _ $t _arithmetic_vector>]() {
                let (l1, l2) = (vec![1 as $t; DATA_SIZE], vec![3 as $t; DATA_SIZE]);
                test_arithmetic_vector_all::<$t, $im>(l1, l2);
            }

            #[test]
            fn [<test_ $im:lower _ $t _cmp_value>]() {
                let (l1, _) = (vec![1 as $t; DATA_SIZE], vec![3 as $t; DATA_SIZE]);
                test_cmp_value_all::<$t, $im>(l1, 2 as $t);
            }

            #[test]
            fn [<test_ $im:lower _ $t _cmp_vector>]() {
                let (l1, l2) = (vec![1 as $t; DATA_SIZE], vec![3 as $t; DATA_SIZE]);
                test_cmp_vector_all::<$t, $im>(l1, l2);
            }
        }
    };
}

fn test_arithmetic_value_all<T: Copy + PartialEq + Debug, R>(l1: Vec<T>, value: T)
where
    R: SimdRegister<T>,
    AutoMath: Math<T>,
    for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
{
    unsafe {
        op_arithmetic_value::tests::test_add::<_, R>(l1.clone(), value);
        op_arithmetic_value::tests::test_sub::<_, R>(l1.clone(), value);
        op_arithmetic_value::tests::test_div::<_, R>(l1.clone(), value);
        op_arithmetic_value::tests::test_mul::<_, R>(l1, value)
    };
}

fn test_arithmetic_vector_all<T: Copy + PartialEq + Debug, R>(l1: Vec<T>, l2: Vec<T>)
where
    R: SimdRegister<T>,
    AutoMath: Math<T>,
    for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
{
    unsafe {
        op_arithmetic_vertical::tests::test_add::<_, R>(l1.clone(), l2.clone());
        op_arithmetic_vertical::tests::test_sub::<_, R>(l1.clone(), l2.clone());
        op_arithmetic_vertical::tests::test_div::<_, R>(l1.clone(), l2.clone());
        op_arithmetic_vertical::tests::test_mul::<_, R>(l1, l2);
    };
}

fn test_cmp_vector_all<T: Copy + PartialEq + Debug, R>(l1: Vec<T>, l2: Vec<T>)
where
    R: SimdRegister<T>,
    AutoMath: Math<T>,
    for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
{
    unsafe {
        op_cmp_vertical::tests::test_eq::<_, R>(l1.clone(), l2.clone());
        op_cmp_vertical::tests::test_neq::<_, R>(l1.clone(), l2.clone());
        op_cmp_vertical::tests::test_lt::<_, R>(l1.clone(), l2.clone());
        op_cmp_vertical::tests::test_lte::<_, R>(l1.clone(), l2.clone());
        op_cmp_vertical::tests::test_gt::<_, R>(l1.clone(), l2.clone());
        op_cmp_vertical::tests::test_gte::<_, R>(l1, l2);
    };
}

fn test_cmp_value_all<T: Copy + PartialEq + Debug, R>(l1: Vec<T>, value: T)
where
    R: SimdRegister<T>,
    AutoMath: Math<T>,
    for<'a> &'a mut [T]: WriteOnlyBuffer<Item = T>,
{
    unsafe {
        op_cmp_value::tests::test_eq::<_, R>(l1.clone(), value);
        op_cmp_value::tests::test_neq::<_, R>(l1.clone(), value);
        op_cmp_value::tests::test_lt::<_, R>(l1.clone(), value);
        op_cmp_value::tests::test_lte::<_, R>(l1.clone(), value);
        op_cmp_value::tests::test_gt::<_, R>(l1.clone(), value);
        op_cmp_value::tests::test_gte::<_, R>(l1, value);
    };
}

test_suite!(f32, Fallback);
test_suite!(f64, Fallback);
test_suite!(i8, Fallback);
test_suite!(i16, Fallback);
test_suite!(i32, Fallback);
test_suite!(i64, Fallback);
test_suite!(u8, Fallback);
test_suite!(u16, Fallback);
test_suite!(u32, Fallback);
test_suite!(u64, Fallback);

test_cosine_extra!(f32, Fallback);
test_cosine_extra!(f64, Fallback);
// test_cosine_extra!(i32, Fallback); - Divide by zero error from RNG on miri.
// test_cosine_extra!(i64, Fallback); - Divide by zero error from RNG on miri.
test_cosine_extra!(u8, Fallback);
test_cosine_extra!(u16, Fallback);
test_cosine_extra!(u32, Fallback);
test_cosine_extra!(u64, Fallback);

test_nan_sanity!(f32, Fallback);
test_nan_sanity!(f64, Fallback);

#[cfg(all(target_feature = "avx2", test))]
mod avx2_tests {
    use super::*;

    test_suite!(f32, Avx2);
    test_suite!(f64, Avx2);
    test_suite!(i8, Avx2);
    test_suite!(i16, Avx2);
    test_suite!(i32, Avx2);
    test_suite!(i64, Avx2);
    test_suite!(u8, Avx2);
    test_suite!(u16, Avx2);
    test_suite!(u32, Avx2);
    test_suite!(u64, Avx2);

    test_cosine_extra!(f32, Avx2);
    test_cosine_extra!(f64, Avx2);
    // test_cosine_extra!(i32, Avx2); - Divide by zero error from RNG on miri.
    // test_cosine_extra!(i64, Avx2); - Divide by zero error from RNG on miri.
    test_cosine_extra!(u8, Avx2);
    test_cosine_extra!(u16, Avx2);
    test_cosine_extra!(u32, Avx2);
    test_cosine_extra!(u64, Avx2);

    test_nan_sanity!(f32, Avx2);
    test_nan_sanity!(f64, Avx2);
}

#[cfg(all(target_feature = "avx512f", feature = "nightly", test))]
mod avx512_tests {
    use super::*;

    test_suite!(f32, Avx512);
    test_suite!(f64, Avx512);
    test_suite!(i8, Avx512);
    test_suite!(i16, Avx512);
    test_suite!(i32, Avx512);
    test_suite!(i64, Avx512);
    test_suite!(u8, Avx512);
    test_suite!(u16, Avx512);
    test_suite!(u32, Avx512);
    test_suite!(u64, Avx512);

    test_cosine_extra!(f32, Avx512);
    test_cosine_extra!(f64, Avx512);
    test_cosine_extra!(i32, Avx512);
    test_cosine_extra!(i64, Avx512);
    test_cosine_extra!(u8, Avx512);
    test_cosine_extra!(u16, Avx512);
    test_cosine_extra!(u32, Avx512);
    test_cosine_extra!(u64, Avx512);

    test_nan_sanity!(f32, Avx512);
    test_nan_sanity!(f64, Avx512);
}

#[cfg(all(target_feature = "avx2", target_feature = "fma", test))]
mod avx2fma_tests {
    use super::*;

    test_suite!(f32, Avx2Fma);
    test_suite!(f64, Avx2Fma);

    test_cosine_extra!(f32, Avx2Fma);
    test_cosine_extra!(f64, Avx2Fma);
}

#[cfg(all(target_feature = "neon", test))]
mod neon_tests {
    use super::*;

    test_suite!(f32, Neon);
    test_suite!(f64, Neon);
    test_suite!(i8, Neon);
    test_suite!(i16, Neon);
    test_suite!(i32, Neon);
    test_suite!(i64, Neon);
    test_suite!(u8, Neon);
    test_suite!(u16, Neon);
    test_suite!(u32, Neon);
    test_suite!(u64, Neon);

    test_cosine_extra!(f32, Neon);
    test_cosine_extra!(f64, Neon);
    test_cosine_extra!(i8, Neon);
    // test_cosine_extra!(i16, Neon); - Divide by zero error from RNG.
    test_cosine_extra!(i32, Neon);
    test_cosine_extra!(i64, Neon);
    test_cosine_extra!(u8, Neon);
    test_cosine_extra!(u16, Neon);
    test_cosine_extra!(u32, Neon);
    test_cosine_extra!(u64, Neon);

    test_nan_sanity!(f32, Neon);
    test_nan_sanity!(f64, Neon);
}
