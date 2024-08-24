use core::mem::MaybeUninit;
use core::any::TypeId;
use std::ops::AddAssign;
use ndarray::{azip, Array, ArrayBase, Data, DimMax, Dimension, Ix0};
use cfavml::buffer::WriteOnlyBuffer;
use cfavml::safe_trait_arithmetic_ops::ArithmeticOps;

use crate::broadcastable_op::{apply_op_with_broadcast_value, apply_op_with_broadcasting};

pub trait AddFast<Rhs=Self> {
    type Output;

    fn fast_add(self, rhs: Rhs) -> Self::Output;
}

pub trait SubFast<Rhs=Self> {
    type Output;

    fn fast_sub(self, rhs: Rhs) -> Self::Output;
}

pub trait MulFast<Rhs=Self> {
    type Output;

    fn fast_mul(self, rhs: Rhs) -> Self::Output;
}

pub trait DivFast<Rhs=Self> {
    type Output;

    fn fast_div(self, rhs: Rhs) -> Self::Output;
}


impl<A, S, D, E> AddFast<ArrayBase<S, E>> for Array<A, D>
where
    A: ArithmeticOps + AddAssign,
    S: Data<Elem=A>,
    D: Dimension + DimMax<E> + 'static,
    E: Dimension + 'static,
    for<'buf> &'buf mut [MaybeUninit<A>]: WriteOnlyBuffer<Item=A>,
{
    type Output = Array<A, <D as DimMax<E>>::Output>;

    fn fast_add(self, rhs: ArrayBase<S, E>) -> Self::Output {
        self.fast_add(&rhs)
    }
}

impl<'a, A, S, D, E> AddFast<&'a ArrayBase<S, E>> for Array<A, D>
where
    A: ArithmeticOps + AddAssign,
    S: Data<Elem=A>,
    D: Dimension + DimMax<E> + 'static,
    E: Dimension + 'static,
    for<'buf> &'buf mut [MaybeUninit<A>]: WriteOnlyBuffer<Item=A>,
{
    type Output = Array<A, <D as DimMax<E>>::Output>;

    fn fast_add(self, rhs: &'a ArrayBase<S, E>) -> Self::Output {
        // Specialization on 0-dim arrays as we have specialized ops that are
        // much faster and more memory efficient.
        let lhs_is_zero_dim = TypeId::of::<D>() == TypeId::of::<Ix0>();
        let rhs_is_zero_dim = TypeId::of::<E>() == TypeId::of::<Ix0>();

        if lhs_is_zero_dim && rhs_is_zero_dim {
            // # Safety
            // We have already checked the type of `D` and `E` are `Ix0` which in reality is a single value
            // so this is safe to transmute, we don't need to worry about strides or similar.
            let mut lhs = unsafe { crate::utils::unlimited_transmute::<Array<A, D>, Array<A, Ix0>>(self) };
            let rhs = unsafe { crate::utils::unlimited_transmute::<&'a ArrayBase<S, E>, &'a ArrayBase<S, Ix0>>(rhs) };
            
            azip!((a in &mut lhs, b in rhs) *a += *b);
            
            return lhs.into_dimensionality::<<D as DimMax<E>>::Output>().unwrap();
        } else if lhs_is_zero_dim {
            // TODO: Extract into trait call instead for ease of operation.
            let broadcast_value = self.iter().next().copied().unwrap();
            unsafe {
                let result = apply_op_with_broadcast_value(
                    rhs.to_owned(),
                    broadcast_value,
                    cfavml::add_value,
                );
                return result.into_dimensionality::<<D as DimMax<E>>::Output>().unwrap()
            }
        } else if rhs_is_zero_dim {
            // TODO: Extract into trait call instead for ease of operation.
            let broadcast_value = rhs.iter().next().copied().unwrap();
            unsafe {
                let result = apply_op_with_broadcast_value(
                    self,
                    broadcast_value,
                    cfavml::add_value,
                );
                return result.into_dimensionality::<<D as DimMax<E>>::Output>().unwrap()
            }
        }

        // # Safety
        // We are passing a cfavml routine into the system which has very strict
        // behaviour around how it reads and writes memory and will only ever read
        // from the slices once before writing and never touching the position again.
        unsafe {
            apply_op_with_broadcasting(
                self,
                rhs.view(),
                cfavml::add_vector
            )
        }
    }
}



