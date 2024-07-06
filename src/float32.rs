//! f32 based ndarray operations
//!
//! The f32 based operations are probably the most 'optimal' of all the types and routines
//! as they were the original basis for this library and largely the driving force behind the
//! patterns you see in the code, since it was 'optimal' for f32s, it is probably good enough
//! for the rest.
//!
//! ## Notes on IEEE compliance:
//!
//! These routines do not strictly abide by the IEEE rules, in particular min/max operations
//! _technically_ can change across architecture in how they handle `NaN`s.
//! Overall I would not use these operations _if_ you _must_ follow the rules.

use cfavml::danger::*;
use ndarray::{Array1, ArrayView1};

use crate::core::AcceleratedLinearOps;

impl<'a> AcceleratedLinearOps<f32> for ArrayView1<'a, f32> {
    fn sum(&self) -> f32 {
        todo!()
    }

    fn mean(&self) -> f32 {
        todo!()
    }

    fn min(&self) -> f32 {
        todo!()
    }

    fn max(&self) -> f32 {
        todo!()
    }

    fn l2(&self) -> f32 {
        todo!()
    }

    fn l2_squared(&self) -> f32 {
        todo!()
    }

    fn dot(&self, other: &Self) -> f32 {
        todo!()
    }

    fn cosine(&self, other: &Self) -> f32 {
        todo!()
    }

    fn euclidean(&self, other: &Self) -> f32 {
        todo!()
    }

    fn euclidean_squared(&self, other: &Self) -> f32 {
        todo!()
    }
}
