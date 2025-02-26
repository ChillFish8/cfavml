//! The core SIMD api providing a generic abstraction over several architectures and various
//! register sizes.
//!
//! This API allows for generally maximum efficiency across the architectures with the only
//! exception being AVX512 which cannot make full use of it's masked loading behaviour, but
//! under testing this did not have a significant impact.
//!
//! In general the API of this trait is not expected to be used directly and is just a light
//! wrapper around various intrinsics with the exception of [DenseLane] and it's helpers
//! providing high throughput operations on the instructions.

#![allow(clippy::missing_safety_doc)]

use core::mem;

#[doc(hidden)]
#[macro_export]
/// Applies a given operation expression over the dense lane for upto 3 args.
macro_rules! apply_dense {
    ($op:expr, $l1:ident) => {{
        DenseLane {
            a: $op($l1.a),
            b: $op($l1.b),
            c: $op($l1.c),
            d: $op($l1.d),
            e: $op($l1.e),
            f: $op($l1.f),
            g: $op($l1.g),
            h: $op($l1.h),
        }
    }};
    ($op:expr, $l1:ident, value = $l2:expr) => {{
        DenseLane {
            a: $op($l1.a, $l2),
            b: $op($l1.b, $l2),
            c: $op($l1.c, $l2),
            d: $op($l1.d, $l2),
            e: $op($l1.e, $l2),
            f: $op($l1.f, $l2),
            g: $op($l1.g, $l2),
            h: $op($l1.h, $l2),
        }
    }};
    ($op:expr, $l1:ident, $l2:ident) => {{
        DenseLane {
            a: $op($l1.a, $l2.a),
            b: $op($l1.b, $l2.b),
            c: $op($l1.c, $l2.c),
            d: $op($l1.d, $l2.d),
            e: $op($l1.e, $l2.e),
            f: $op($l1.f, $l2.f),
            g: $op($l1.g, $l2.g),
            h: $op($l1.h, $l2.h),
        }
    }};
    ($op:expr, $l1:ident, $l2:ident, $l3:ident) => {{
        DenseLane {
            a: $op($l1.a, $l2.a, $l3.a),
            b: $op($l1.b, $l2.b, $l3.b),
            c: $op($l1.c, $l2.c, $l3.c),
            d: $op($l1.d, $l2.d, $l3.d),
            e: $op($l1.e, $l2.e, $l3.e),
            f: $op($l1.f, $l2.f, $l3.f),
            g: $op($l1.g, $l2.g, $l3.g),
            h: $op($l1.h, $l2.h, $l3.h),
        }
    }};
}

#[derive(Copy, Clone)]
/// A dense lane is formed of `NUM_LANES` smaller SIMD registers.
///
/// The aim of this type is to generally maximize the instruction throughput
/// across any platform and arch.
pub struct DenseLane<T> {
    pub a: T,
    pub b: T,
    pub c: T,
    pub d: T,
    pub e: T,
    pub f: T,
    pub g: T,
    pub h: T,
}

impl<T: Copy> DenseLane<T> {
    /// The number of lanes within the dense lane.
    pub const NUM_LANES: usize = 8;

    #[inline(always)]
    /// Copies the register in `value` to all lanes.
    pub fn copy(value: T) -> Self {
        Self {
            a: value,
            b: value,
            c: value,
            d: value,
            e: value,
            f: value,
            g: value,
            h: value,
        }
    }
}

/// A set of core SIMD operations over the given type.
pub trait SimdRegister<T: Copy> {
    /// The single register for the given arch.
    ///
    /// This is normally something like a [core::arch::x86_64::__m256] or similar.
    type Register: Copy;

    #[inline(always)]
    /// The number of elements `T` in a dense lane.
    fn elements_per_dense() -> usize {
        let num_elements_per_lane = Self::elements_per_lane();
        num_elements_per_lane * DenseLane::<Self::Register>::NUM_LANES
    }

    #[inline(always)]
    /// The number of elements `T` in a dense lane.
    fn elements_per_lane() -> usize {
        mem::size_of::<Self::Register>() / mem::size_of::<T>()
    }

    /// Loads `Self::elements_per_lane` elements of `T` into a `Self::Register`.
    unsafe fn load(mem: *const T) -> Self::Register;

    /// Loads `Self::elements_per_lane` elements of `value` into a `Self::Register`.
    unsafe fn filled(value: T) -> Self::Register;

    /// Creates a new zeroed register.
    unsafe fn zeroed() -> Self::Register;

    #[allow(clippy::identity_op)]
    #[allow(clippy::erasing_op)]
    #[inline(always)]
    /// Loads `Self::element_per_dense` elements of `T` into a `DenseLane<Self::Register>`.
    unsafe fn load_dense(mem: *const T) -> DenseLane<Self::Register> {
        DenseLane {
            a: Self::load(mem.add(Self::elements_per_lane() * 0)),
            b: Self::load(mem.add(Self::elements_per_lane() * 1)),
            c: Self::load(mem.add(Self::elements_per_lane() * 2)),
            d: Self::load(mem.add(Self::elements_per_lane() * 3)),
            e: Self::load(mem.add(Self::elements_per_lane() * 4)),
            f: Self::load(mem.add(Self::elements_per_lane() * 5)),
            g: Self::load(mem.add(Self::elements_per_lane() * 6)),
            h: Self::load(mem.add(Self::elements_per_lane() * 7)),
        }
    }

    #[inline(always)]
    /// Loads `Self::element_per_dense` elements of `T` into a `DenseLane<Self::Register>`.
    unsafe fn filled_dense(value: T) -> DenseLane<Self::Register> {
        DenseLane::copy(Self::filled(value))
    }

    #[inline(always)]
    /// Creates a zeroed dense lane.
    unsafe fn zeroed_dense() -> DenseLane<Self::Register> {
        DenseLane::copy(Self::zeroed())
    }

    /// Perform a element wise add on two dense lanes.
    unsafe fn add(l1: Self::Register, l2: Self::Register) -> Self::Register;

    /// Perform a element wise add on two dense lanes.
    unsafe fn sub(l1: Self::Register, l2: Self::Register) -> Self::Register;

    /// Perform a element wise multiplication on two dense lanes.
    unsafe fn mul(l1: Self::Register, l2: Self::Register) -> Self::Register;

    /// Perform a element wise add on two dense lanes.
    unsafe fn div(l1: Self::Register, l2: Self::Register) -> Self::Register;

    /// Perform a element wise multiply add operations on two dense lanes with an accumulator.
    unsafe fn fmadd(
        l1: Self::Register,
        l2: Self::Register,
        acc: Self::Register,
    ) -> Self::Register;

    /// Perform a element wise max operations on two dense lanes.
    unsafe fn max(l1: Self::Register, l2: Self::Register) -> Self::Register;

    /// Perform a element wise min operations on two dense lanes.
    unsafe fn min(l1: Self::Register, l2: Self::Register) -> Self::Register;

    /// Compares each element in `l1` and `l2` and returns a mask indicating if they are equal.
    unsafe fn eq(l1: Self::Register, l2: Self::Register) -> Self::Register;

    /// Compares each element in `l1` and `l2` and returns a mask indicating if they are not equal.
    unsafe fn neq(l1: Self::Register, l2: Self::Register) -> Self::Register;

    /// Performs a comparison of registers `l1` and `l2` returning a mask of if elements
    /// in `l1` are _less than_ elements in `l2`.
    unsafe fn lt(l1: Self::Register, l2: Self::Register) -> Self::Register;

    /// Performs a comparison of registers `l1` and `l2` returning a mask of if elements
    /// in `l1` are _less than or equal to_ elements in `l2`.
    unsafe fn lte(l1: Self::Register, l2: Self::Register) -> Self::Register;

    /// Performs a comparison of registers `l1` and `l2` returning a mask of if elements
    /// in `l1` are _greater than_ elements in `l2`.
    unsafe fn gt(l1: Self::Register, l2: Self::Register) -> Self::Register;

    /// Performs a comparison of registers `l1` and `l2` returning a mask of if elements
    /// in `l1` are _greater than or equal to_ elements in `l2`.
    unsafe fn gte(l1: Self::Register, l2: Self::Register) -> Self::Register;

    #[inline(always)]
    /// Perform a element wise add on two dense lanes.
    unsafe fn add_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        apply_dense!(Self::add, l1, l2)
    }

    #[inline(always)]
    /// Perform a element wise add on two dense lanes.
    unsafe fn sub_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        apply_dense!(Self::sub, l1, l2)
    }

    #[inline(always)]
    /// Perform a element wise multiplication on two dense lanes.
    unsafe fn mul_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        apply_dense!(Self::mul, l1, l2)
    }

    #[inline(always)]
    /// Perform a element wise add on two dense lanes.
    unsafe fn div_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        apply_dense!(Self::div, l1, l2)
    }

    #[inline(always)]
    /// Perform a element wise multiply add operations on two dense lanes with an accumulator.
    unsafe fn fmadd_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
        acc: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        apply_dense!(Self::fmadd, l1, l2, acc)
    }

    #[inline(always)]
    /// Perform a element wise max operations on two dense lanes.
    unsafe fn max_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        apply_dense!(Self::max, l1, l2)
    }

    #[inline(always)]
    /// Perform a element wise min operations on two dense lanes.
    unsafe fn min_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        apply_dense!(Self::min, l1, l2)
    }

    #[inline(always)]
    /// Compares each element in `l1` and `l2` dense lanes and returns a mask
    /// indicating if  they are equal.
    unsafe fn eq_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        apply_dense!(Self::eq, l1, l2)
    }

    #[inline(always)]
    /// Compares each element in `l1` and `l2` dense lanes and returns a mask
    /// indicating if they are not equal.
    unsafe fn neq_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        apply_dense!(Self::neq, l1, l2)
    }

    #[inline(always)]
    /// Performs a comparison of dense lanes `l1` and `l2` returning a mask of if elements
    /// in `l1` are _less than_ elements in `l2`.
    unsafe fn lt_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        apply_dense!(Self::lt, l1, l2)
    }

    #[inline(always)]
    /// Performs a comparison of dense lanes `l1` and `l2` returning a mask of if elements
    /// in `l1` are _less than or equal to_ elements in `l2`.
    unsafe fn lte_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        apply_dense!(Self::lte, l1, l2)
    }

    #[inline(always)]
    /// Performs a comparison of dense lanes `l1` and `l2` returning a mask of if elements
    /// in `l1` are _greater than_ elements in `l2`.
    unsafe fn gt_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        apply_dense!(Self::gt, l1, l2)
    }

    #[inline(always)]
    /// Performs a comparison of dense lanes `l1` and `l2` returning a mask of if elements
    /// in `l1` are _greater than or equal to_ elements in `l2`.
    unsafe fn gte_dense(
        l1: DenseLane<Self::Register>,
        l2: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        apply_dense!(Self::gte, l1, l2)
    }

    /// Performs a horizontal sum of the register returning the resulting value `T`.
    unsafe fn sum_to_value(reg: Self::Register) -> T;

    #[inline(always)]
    /// Rolls up a dense lane into a single register.
    unsafe fn sum_to_register(lane: DenseLane<Self::Register>) -> Self::Register {
        let mut acc1 = Self::add(lane.a, lane.b);
        let acc2 = Self::add(lane.c, lane.d);
        let mut acc3 = Self::add(lane.e, lane.f);
        let acc4 = Self::add(lane.g, lane.h);

        acc1 = Self::add(acc1, acc2);
        acc3 = Self::add(acc3, acc4);

        Self::add(acc1, acc3)
    }

    /// Performs a horizontal max of the register returning the resulting value `T`.
    unsafe fn max_to_value(reg: Self::Register) -> T;

    #[inline(always)]
    /// Does an element wise max of a dense lane into a single register.
    unsafe fn max_to_register(lane: DenseLane<Self::Register>) -> Self::Register {
        let mut acc1 = Self::max(lane.a, lane.b);
        let acc2 = Self::max(lane.c, lane.d);
        let mut acc3 = Self::max(lane.e, lane.f);
        let acc4 = Self::max(lane.g, lane.h);

        acc1 = Self::max(acc1, acc2);
        acc3 = Self::max(acc3, acc4);

        Self::max(acc1, acc3)
    }

    /// Performs a horizontal max of the register returning the resulting value `T`.
    unsafe fn min_to_value(reg: Self::Register) -> T;

    #[inline(always)]
    /// Does an element wise min of a dense lane into a single register.
    unsafe fn min_to_register(lane: DenseLane<Self::Register>) -> Self::Register {
        let mut acc1 = Self::min(lane.a, lane.b);
        let acc2 = Self::min(lane.c, lane.d);
        let mut acc3 = Self::min(lane.e, lane.f);
        let acc4 = Self::min(lane.g, lane.h);

        acc1 = Self::min(acc1, acc2);
        acc3 = Self::min(acc3, acc4);

        Self::min(acc1, acc3)
    }

    /// Writes a single register to the given memory.
    ///
    /// Writes `mem::size_of::<Self::Register>() / mem::size_of::<T>()` elements to the pointer.
    unsafe fn write(mem: *mut T, reg: Self::Register);

    #[allow(clippy::identity_op)]
    #[allow(clippy::erasing_op)]
    #[inline(always)]
    /// Write a dense lane to the given memory.
    ///
    /// This writes `Self::elements_size` number of elements to the pointer.
    unsafe fn write_dense(mem: *mut T, lane: DenseLane<Self::Register>) {
        Self::write(mem.add(Self::elements_per_lane() * 0), lane.a);
        Self::write(mem.add(Self::elements_per_lane() * 1), lane.b);
        Self::write(mem.add(Self::elements_per_lane() * 2), lane.c);
        Self::write(mem.add(Self::elements_per_lane() * 3), lane.d);
        Self::write(mem.add(Self::elements_per_lane() * 4), lane.e);
        Self::write(mem.add(Self::elements_per_lane() * 5), lane.f);
        Self::write(mem.add(Self::elements_per_lane() * 6), lane.g);
        Self::write(mem.add(Self::elements_per_lane() * 7), lane.h);
    }
}

pub trait Hypot<T>: SimdRegister<T>
where
    T: Copy,
{
    /// SIMD Variant of the std hypot function. Computes the distance between the origin
    /// and a point (`x`, `y`) on the Euclidean plane.
    unsafe fn hypot(a: Self::Register, b: Self::Register) -> Self::Register;
    #[inline(always)]

    /// Perform a element wise hypot on two dense lanes.
    unsafe fn hypot_dense(
        a: DenseLane<Self::Register>,
        b: DenseLane<Self::Register>,
    ) -> DenseLane<Self::Register> {
        apply_dense!(Self::hypot, a, b)
    }
}
