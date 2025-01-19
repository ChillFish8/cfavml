use core::any::TypeId;
use core::mem::MaybeUninit;

use cfavml::buffer::WriteOnlyBuffer;
use ndarray::{azip, Array, ArrayBase, Data, DimMax, Dimension, Ix0};

use crate::broadcastable_op::{
    apply_op_with_broadcast_value,
    apply_op_with_broadcasting,
};

pub trait AddFast<Rhs = Self> {
    type Output;

    fn fast_add(self, rhs: Rhs) -> Self::Output;
}

pub trait SubFast<Rhs = Self> {
    type Output;

    fn fast_sub(self, rhs: Rhs) -> Self::Output;
}

pub trait MulFast<Rhs = Self> {
    type Output;

    fn fast_mul(self, rhs: Rhs) -> Self::Output;
}

pub trait DivFast<Rhs = Self> {
    type Output;

    fn fast_div(self, rhs: Rhs) -> Self::Output;
}

macro_rules! implement_op_for_type {
    ($t:ty, $imp:ident, $trait_fn:ident, $cfavml_op:ident, $single_value_step:tt) => {
        impl<D> $imp<$t> for Array<$t, D>
        where
            D: Dimension + 'static,
            for<'buf> &'buf mut [MaybeUninit<$t>]: WriteOnlyBuffer<Item=$t>,
        {
            type Output = Array<$t, D>;

            fn $trait_fn(self, rhs: $t) -> Self::Output {
                unsafe {
                    apply_op_with_broadcast_value(
                        self,
                        rhs,
                        |rhs, lhs, w| cfavml::$cfavml_op(lhs, rhs, w)
                    )
                }
            }
        }

        impl<D> $imp<Array<$t, D>> for $t
        where
            D: Dimension + 'static,
            for<'buf> &'buf mut [MaybeUninit<$t>]: WriteOnlyBuffer<Item=$t>,
        {
            type Output = Array<$t, D>;

            fn $trait_fn(self, rhs: Array<$t, D>) -> Self::Output {
                unsafe {
                    apply_op_with_broadcast_value(
                        rhs,
                        self,
                        |lhs, rhs, w| cfavml::$cfavml_op(lhs, rhs, w)
                    )
                }
            }
        }

        impl<S, D, E> $imp<ArrayBase<S, E>> for Array<$t, D>
        where
            S: Data<Elem=$t>,
            D: Dimension + DimMax<E> + 'static,
            E: Dimension + 'static,
            for<'buf> &'buf mut [MaybeUninit<$t>]: WriteOnlyBuffer<Item=$t>,
        {
            type Output = Array<$t, <D as DimMax<E>>::Output>;

            fn $trait_fn(self, rhs: ArrayBase<S, E>) -> Self::Output {
                self.$trait_fn(&rhs)
            }
        }

        impl<'a, S, D, E> $imp<&'a ArrayBase<S, E>> for Array<$t, D>
        where
            S: Data<Elem=$t>,
            D: Dimension + DimMax<E> + 'static,
            E: Dimension + 'static,
            for<'buf> &'buf mut [MaybeUninit<$t>]: WriteOnlyBuffer<Item=$t>,
        {
            type Output = Array<$t, <D as DimMax<E>>::Output>;

            fn $trait_fn(self, rhs: &'a ArrayBase<S, E>) -> Self::Output {
                // Specialization on 0-dim arrays as we have specialized ops that are
                // much faster and more memory efficient.
                let lhs_is_zero_dim = TypeId::of::<D>() == TypeId::of::<Ix0>();
                let rhs_is_zero_dim = TypeId::of::<E>() == TypeId::of::<Ix0>();

                if lhs_is_zero_dim && rhs_is_zero_dim {
                    // # Safety
                    // We have already checked the type of `D` and `E` are `Ix0` which in reality is a single value
                    // so this is safe to transmute, we don't need to worry about strides or similar.
                    let mut lhs = unsafe { crate::utils::unlimited_transmute::<Array<$t, D>, Array<$t, Ix0>>(self) };
                    let rhs = unsafe { crate::utils::unlimited_transmute::<&'a ArrayBase<S, E>, &'a ArrayBase<S, Ix0>>(rhs) };

                    azip!((a in &mut lhs, b in rhs) *a $single_value_step *b);

                    return lhs.into_dimensionality::<<D as DimMax<E>>::Output>().unwrap();
                } else if lhs_is_zero_dim {
                    let broadcast_value = self.iter().next().copied().unwrap();
                    let result = broadcast_value.$trait_fn(rhs.to_owned());
                    return result.into_dimensionality::<<D as DimMax<E>>::Output>().unwrap()
                } else if rhs_is_zero_dim {
                    let broadcast_value = rhs.iter().next().copied().unwrap();
                    let result = self.$trait_fn(broadcast_value);
                    return result.into_dimensionality::<<D as DimMax<E>>::Output>().unwrap()
                }

                // # Safety
                // We are passing a cfavml routine into the system which has very strict
                // behaviour around how it reads and writes memory and will only ever read
                // from the slices once before writing and never touching the position again.
                unsafe {
                    apply_op_with_broadcasting(
                        self,
                        rhs.view(),
                        |a, b, w| cfavml::$cfavml_op(a, b, w)
                    )
                }
            }
        }
    };
}

macro_rules! impl_core_ops_for_type {
    ($t:ty) => {
        implement_op_for_type!($t, AddFast, fast_add, add_vertical, +=);
        implement_op_for_type!($t, SubFast, fast_sub, sub_vertical, -=);
        implement_op_for_type!($t, MulFast, fast_mul, mul_vertical, *=);
        implement_op_for_type!($t, DivFast, fast_div, div_vertical, /=);
    };
}

impl_core_ops_for_type!(f32);
impl_core_ops_for_type!(f64);
impl_core_ops_for_type!(i8);
impl_core_ops_for_type!(i16);
impl_core_ops_for_type!(i32);
impl_core_ops_for_type!(i64);
impl_core_ops_for_type!(u8);
impl_core_ops_for_type!(u16);
impl_core_ops_for_type!(u32);
impl_core_ops_for_type!(u64);

#[inline]
/// Performs a vertical addition of the LHS and RHS values.
///
/// Inputs can be in the form of two arrays or some combination of
/// one array and a broadcast value.
///
/// ### Broadcasting
///
/// Broadcasting behaves the same as the rest of the ndarray operations
/// which intern try and mimic how numpy broadcasts arrays.
/// Unfortunately performance when broadcasting may be sub-par due to
/// unnecessary allocations.
///
/// ### Note on wrapping
///
/// The nature of these SIMD routines means they _always_ wrap, they will never
/// panic in debug builds around overflows unlike the default in ndarray.
pub fn add<Lhs, Rhs>(lhs: Lhs, rhs: Rhs) -> Lhs::Output
where
    Lhs: AddFast<Rhs>,
{
    lhs.fast_add(rhs)
}

#[inline]
/// Performs a vertical subtraction of the LHS and RHS values.
///
/// Inputs can be in the form of two arrays or some combination of
/// one array and a broadcast value.
///
/// ### Broadcasting
///
/// Broadcasting behaves the same as the rest of the ndarray operations
/// which intern try and mimic how numpy broadcasts arrays.
/// Unfortunately performance when broadcasting may be sub-par due to
/// unnecessary allocations.
///
/// ### Note on wrapping
///
/// The nature of these SIMD routines means they _always_ wrap, they will never
/// panic in debug builds around overflows unlike the default in ndarray.
pub fn sub<Lhs, Rhs>(lhs: Lhs, rhs: Rhs) -> Lhs::Output
where
    Lhs: SubFast<Rhs>,
{
    lhs.fast_sub(rhs)
}

#[inline]
/// Performs a vertical multiply of the LHS and RHS values.
///
/// Inputs can be in the form of two arrays or some combination of
/// one array and a broadcast value.
///
/// ### Broadcasting
///
/// Broadcasting behaves the same as the rest of the ndarray operations
/// which intern try and mimic how numpy broadcasts arrays.
/// Unfortunately performance when broadcasting may be sub-par due to
/// unnecessary allocations.
///
/// ### Note on wrapping
///
/// The nature of these SIMD routines means they _always_ wrap, they will never
/// panic in debug builds around overflows unlike the default in ndarray.
pub fn mul<Lhs, Rhs>(lhs: Lhs, rhs: Rhs) -> Lhs::Output
where
    Lhs: MulFast<Rhs>,
{
    lhs.fast_mul(rhs)
}

#[inline]
/// Performs a vertical division of the LHS and RHS values.
///
/// Inputs can be in the form of two arrays or some combination of
/// one array and a broadcast value.
///
/// ### Broadcasting
///
/// Broadcasting behaves the same as the rest of the ndarray operations
/// which intern try and mimic how numpy broadcasts arrays.
/// Unfortunately performance when broadcasting may be sub-par due to
/// unnecessary allocations.
///
/// ### Note on wrapping
///
/// The nature of these SIMD routines means they _always_ wrap, they will never
/// panic in debug builds around overflows unlike the default in ndarray.
pub fn div<Lhs, Rhs>(lhs: Lhs, rhs: Rhs) -> Lhs::Output
where
    Lhs: DivFast<Rhs>,
{
    lhs.fast_div(rhs)
}

#[cfg(test)]
mod tests {
    use ndarray::{Array0, Array3, ArrayView3};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    use super::*;

    macro_rules! impl_arithmetic_tests {
        ($t:ty, $op:ident, $manual_op:tt) => {
            paste::paste! {
                #[test]
                fn [< test_one_array_lhs_broadcast_value_ $t _ $op >]() {
                    let a = Array3::random((3, 3, 3), Uniform::new(1 as $t, 10 as $t));

                    let expected: Array<$t, _> = 1 as $t $manual_op &a;
                    let output: Array<$t, _> = $op(1 as $t, a);
                    assert_eq!(output.shape(), expected.shape());
                    assert_eq!(output, expected);
                }

                #[test]
                fn [< test_one_array_rhs_broadcast_value_ $t _ $op >]() {
                    let a = Array3::random((3, 3, 3), Uniform::new(1 as $t, 10 as $t));

                    let expected: Array<$t, _> = &a $manual_op 1  as $t;
                    let output: Array<$t, _> = $op(a, 1 as $t);
                    assert_eq!(output.shape(), expected.shape());
                    assert_eq!(output, expected);
                }

                #[test]
                fn [< test_two_zerodim_ $t _ $op >]() {
                    let a = Array0::random((), Uniform::new(1 as $t, 10 as $t));
                    let b = Array0::random((), Uniform::new(1 as $t, 10 as $t));

                    let expected: Array<$t, _> = &a $manual_op &b;
                    let output: Array<$t, _> = $op(a, b);
                    assert_eq!(output.shape(), expected.shape());
                    assert_eq!(output, expected);
                }

                #[test]
                fn [< test_one_array_lhs_zerodim_ $t _ $op >]() {
                    let a = Array3::random((3, 3, 3), Uniform::new(1 as $t, 10 as $t));
                    let b = Array0::random((), Uniform::new(1 as $t, 10 as $t));

                    let expected: Array<$t, _> = &b $manual_op &a;
                    let output: Array<$t, _> = $op(b, a);
                    assert_eq!(output.shape(), expected.shape());
                    assert_eq!(output, expected);
                }

                #[test]
                fn [< test_one_array_rhs_zerodim_ $t _ $op >]() {
                    let a = Array3::random((3, 3, 3), Uniform::new(1 as $t, 10 as $t));
                    let b = Array0::random((), Uniform::new(1 as $t, 10 as $t));

                    let expected: Array<$t, _> = &a $manual_op &b;
                    let output: Array<$t, _> = $op(a, b);
                    assert_eq!(output.shape(), expected.shape());
                    assert_eq!(output, expected);
                }

                #[test]
                fn [< test_two_arrays_no_broadcast_ $t _ $op >]() {
                    let a = Array3::random((3, 3, 3), Uniform::new(1 as $t, 10 as $t));
                    let b = Array3::random((3, 3, 3), Uniform::new(1 as $t, 10 as $t));

                    let expected = &a $manual_op &b;
                    let output: Array<$t, _> = $op(a, b);
                    assert_eq!(output.shape(), expected.shape());
                    assert_eq!(output, expected);
                }

                #[test]
                fn [< test_two_arrays_owned_broadcast_ $t _ $op >]() {
                    let a = Array3::random((3, 3, 3), Uniform::new(1 as $t, 10 as $t));
                    let b = Array3::random((1, 3, 3), Uniform::new(1 as $t, 10 as $t));

                    let expected = &a $manual_op &b;
                    let output: Array<$t, _> = $op(a, b);
                    assert_eq!(output.shape(), expected.shape());
                    assert_eq!(output, expected);
                }

                #[test]
                fn [< test_two_arrays_slice_broadcast_ $t _ $op >]() {
                    let a = Array3::random((3, 3, 3), Uniform::new(1 as $t, 10 as $t));
                    let b = ArrayView3::from_shape((1, 3, 3), &[1 as $t, 2 as $t, 3 as $t, 1 as $t, 2 as $t, 3 as $t, 1 as $t, 2 as $t, 3 as $t]).unwrap();

                    let expected = &a $manual_op &b;
                    let output: Array<$t, _> = $op(a, b);
                    assert_eq!(output.shape(), expected.shape());
                    assert_eq!(output, expected);
                }

                #[test]
                fn [< test_two_arrays_twin_stretch_broadcast_ $t _ $op >]() {
                    let a = Array3::random((3, 3, 1), Uniform::new(1 as $t, 10 as $t));
                    let b = Array3::random((1, 3, 3), Uniform::new(1 as $t, 10 as $t));

                    let expected = &a $manual_op &b;
                    let output: Array<$t, _> = $op(a, b);
                    assert_eq!(output.shape(), expected.shape());
                    assert_eq!(output, expected);
                }
            }
        };
    }

    macro_rules! impl_ops_tests_for_type {
        ($t:ty) => {
            impl_arithmetic_tests!($t, add, +);
            impl_arithmetic_tests!($t, sub, -);
            impl_arithmetic_tests!($t, mul, *);
            impl_arithmetic_tests!($t, div, /);
        };
    }

    impl_ops_tests_for_type!(f32);
    impl_ops_tests_for_type!(f64);
    impl_ops_tests_for_type!(i8);
    impl_ops_tests_for_type!(i16);
    impl_ops_tests_for_type!(i32);
    impl_ops_tests_for_type!(i64);
    impl_ops_tests_for_type!(u8);
    impl_ops_tests_for_type!(u16);
    impl_ops_tests_for_type!(u32);
    impl_ops_tests_for_type!(u64);
}
