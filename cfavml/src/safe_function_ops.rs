//! All CFAVML routines exposed as generic functions.
//!
//! These functions use runtime dispatch when the `std` feature is enabled, otherwise in
//! no std environments, compile time dispatch is used by detecting the `target_feature` cfg.
//!
//! This means you when runtime detection is disabled, you must compile with one of `target-cpu`
//! or `target-feature` Rust flags set otherwise this will always use the `Fallback` implementations.

use crate::buffer::WriteOnlyBuffer;
use crate::mem_loader::{IntoMemLoader, MemLoader};
use crate::safe_trait_agg_ops::AggOps;
use crate::safe_trait_arithmetic_ops::ArithmeticOps;
use crate::safe_trait_cmp_ops::CmpOps;
use crate::safe_trait_distance_ops::DistanceOps;

#[inline]
/// Calculates the cosine similarity distance of vectors `a` and `b`.
///
/// ### Pseudocode
///
/// ```ignore
/// result = 0
/// norm_a = 0
/// norm_b = 0
///
/// for i in range(dims):
///     result += a[i] * b[i]
///     norm_a += a[i] ** 2
///     norm_b += b[i] ** 2
///
/// if norm_a == 0.0 and norm_b == 0.0:
///     return 0.0
/// elif norm_a == 0.0 or norm_b == 0.0:
///     return 1.0
/// else:
///     return 1.0 - (result / sqrt(norm_a * norm_b))
/// ```
///
/// ### Panics
///
/// This function will panic if vectors `a` and `b` do not match in size.
pub fn cosine<T, B1, B2>(a: B1, b: B2) -> T
where 
    T: DistanceOps,
    B1: IntoMemLoader<T>,
    B1::Loader: MemLoader<Value=T>,
    B2: IntoMemLoader<T>,
    B2::Loader: MemLoader<Value=T>,
{
    T::cosine(a, b)
}

#[inline]
/// Calculates the cosine similarity distance of vectors `a` and `b`.
///
/// ### Pseudocode
///
/// ```ignore
/// result = 0
///
/// for i in range(dims):
///     result += a[i] * b[i]
///
/// return result
/// ```
///
/// ### Panics
///
/// This function will panic if vectors `a` and `b` do not match in size.
pub fn dot<T, B1, B2>(a: B1, b: B2) -> T
where
    T: DistanceOps,
    B1: IntoMemLoader<T>,
    B1::Loader: MemLoader<Value=T>,
    B2: IntoMemLoader<T>,
    B2::Loader: MemLoader<Value=T>,
{
    T::dot(a, b)
}

#[inline]
/// Calculates the squared Euclidean distance of vectors `a` and `b`.
///
/// ### Pseudocode
///
/// ```ignore
/// result = 0
///
/// for i in range(dims):
///     diff = a[i] - b[i]
///     result += diff * diff
///
/// return result
/// ```
///
/// ### Panics
///
/// This function will panic if vectors `a` and `b` do not match in size.
pub fn squared_euclidean<T, B1, B2>(a: B1, b: B2) -> T
where
    T: DistanceOps,
    B1: IntoMemLoader<T>,
    B1::Loader: MemLoader<Value=T>,
    B2: IntoMemLoader<T>,
    B2::Loader: MemLoader<Value=T>,
{
    T::squared_euclidean(a, b)
}

#[inline]
/// Calculates the squared L2 norm of vector `a`.
///
/// ### Pseudocode
///
/// ```ignore
/// result = 0
///
/// for i in range(dims):
///     result += a[i] * a[i]
///
/// return result
/// ```
pub fn squared_norm<T, B1>(a: B1) -> T
where
    T: DistanceOps,
    B1: IntoMemLoader<T>,
    B1::Loader: MemLoader<Value=T>,
{
    T::squared_norm(a)
}

#[inline]
/// Performs a horizontal sum of all elements in a returning the result.
///
/// ### Pseudocode
///
/// ```ignore
/// result = 0
///
/// for i in range(dims):
///     result += a[i]
///
/// return result
/// ```
pub fn sum<T, B1>(a: B1) -> T
where
    T: AggOps,
    B1: IntoMemLoader<T>,
    B1::Loader: MemLoader<Value=T>,
{
    T::sum(a)
}

#[inline]
/// Finds the horizontal max element of a given vector and returns the result.
///
/// ### Pseudocode
///
/// ```ignore
/// result = -inf
///
/// for i in range(dims):
///     result = max(result, a[i])
///
/// return result
/// ```
pub fn max<T, B1>(a: B1) -> T
where
    T: CmpOps,
    B1: IntoMemLoader<T>,
    B1::Loader: MemLoader<Value=T>,
{
    T::max(a)
}

#[inline]
/// Performs an element wise max on each element pair from vectors `a` and `b`, writing the result
/// to `result`.
///
/// ### Pseudocode
///
/// ```ignore
/// result = [0; dims]
///
/// for i in range(dims):
///     result[i] = max(a[i], b[i])
///
/// return result
/// ```
///
/// ### Result buffer
///
/// The result buffer can be either an initialized slice i.e. `&mut [T]`
/// or it can be a slice holding potentially uninitialized data i.e. `&mut [MaybeUninit<T>]`.
///
/// Once the operation is complete, it is safe to assume the data written is fully initialized.
///
/// ### Panics
///
/// Panics if the size of vectors `a`, `b` and `result` do not match.
pub fn max_vertical<T, B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
where
    T: CmpOps,
    B1: IntoMemLoader<T>,
    B1::Loader: MemLoader<Value = T>,
    B2: IntoMemLoader<T>,
    B2::Loader: MemLoader<Value = T>,
    for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = T>,
{
    T::max_vertical(lhs, rhs, result)
}

#[inline]
/// Finds the horizontal min element of a given vector and returns the result.
///
/// ### Pseudocode
///
/// ```ignore
/// result = inf
///
/// for i in range(dims):
///     result = min(result, a[i])
///
/// return result
/// ```
pub fn min<T, B1>(a: B1) -> T
where
    T: CmpOps,
    B1: IntoMemLoader<T>,
    B1::Loader: MemLoader<Value=T>,
{
    T::min(a)
}

#[inline]
/// Performs an element wise min on each element pair from vectors `a` and `b`, writing the result
/// to `result`.
///
/// ### Pseudocode
///
/// ```ignore
/// result = [0; dims]
///
/// for i in range(dims):
///     result[i] = min(a[i], b[i])
///
/// return result
/// ```
///
/// ### Result buffer
///
/// The result buffer can be either an initialized slice i.e. `&mut [T]`
/// or it can be a slice holding potentially uninitialized data i.e. `&mut [MaybeUninit<T>]`.
///
/// Once the operation is complete, it is safe to assume the data written is fully initialized.
///
/// ### Panics
///
/// Panics if the size of vectors `a`, `b` and `result` do not match.
pub fn min_vertical<T, B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
where
    T: CmpOps,
    B1: IntoMemLoader<T>,
    B1::Loader: MemLoader<Value = T>,
    B2: IntoMemLoader<T>,
    B2::Loader: MemLoader<Value = T>,
    for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = T>,
{
    T::min_vertical(lhs, rhs, result)
}

#[inline]
/// Checks each element pair from vectors `a` and `b` of size `dims`  comparing
/// if element `a` is **_equal to_** element `b` returning a mask vector of the same type.
///
/// ### Pseudocode
///
/// ```ignore
/// mask = [0; dims]
///
/// for i in range(dims):
///     mask[i] = a[i] == b[i] ? 1 : 0
///
/// return mask
/// ```
///
/// ### Note on `NaN` handling on `f32/f64` types
///
/// For `f32` and `f64` types, `NaN` values are handled as always being `false` in **ANY** comparison.
/// Even when compared against each other.
///
/// - `0.0 == 0.0 -> true`
/// - `0.0 == NaN -> false`
/// - `NaN == NaN -> false`
///
/// ### Result buffer
///
/// The result buffer can be either an initialized slice i.e. `&mut [T]`
/// or it can be a slice holding potentially uninitialized data i.e. `&mut [MaybeUninit<T>]`.
///
/// Once the operation is complete, it is safe to assume the data written is fully initialized.
///
/// ### Panics
///
/// Panics if the size of vectors `a`, `b` and `result` do not match.
pub fn eq_vertical<T, B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
where
    T: CmpOps,
    B1: IntoMemLoader<T>,
    B1::Loader: MemLoader<Value = T>,
    B2: IntoMemLoader<T>,
    B2::Loader: MemLoader<Value = T>,
    for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = T>,
{
    T::eq_vertical(lhs, rhs, result)
}


#[inline]
/// Checks each element pair from vectors `a` and `b` of size `dims`  comparing
/// if element `a` is **_not equal to_** element `b` returning a mask vector of the same type.
///
/// ### Pseudocode
///
/// ```ignore
/// mask = [0; dims]
///
/// for i in range(dims):
///     mask[i] = a[i] != b[i] ? 1 : 0
///
/// return mask
/// ```
///
/// ### Note on `NaN` handling on `f32/f64` types
///
/// For `f32` and `f64` types, `NaN` values are handled as always being `false` in **ANY** comparison.
/// Even when compared against each other, meaning in the case of NOT equal, they become `true`.
///
/// - `0.0 != 1.0 -> true`
/// - `0.0 != NaN -> true`
/// - `NaN != NaN -> true`
///
/// ### Result buffer
///
/// The result buffer can be either an initialized slice i.e. `&mut [T]`
/// or it can be a slice holding potentially uninitialized data i.e. `&mut [MaybeUninit<T>]`.
///
/// Once the operation is complete, it is safe to assume the data written is fully initialized.
///
/// ### Panics
///
/// Panics if the size of vectors `a`, `b` and `result` do not match.
pub fn neq_vertical<T, B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
where
    T: CmpOps,
    B1: IntoMemLoader<T>,
    B1::Loader: MemLoader<Value = T>,
    B2: IntoMemLoader<T>,
    B2::Loader: MemLoader<Value = T>,
    for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = T>,
{
    T::neq_vertical(lhs, rhs, result)
}

#[inline]
/// Checks each element pair from vectors `a` and `b` of size `dims`  comparing
/// if element `a` is **_less than_** element `b` returning a mask vector of the same type.
///
/// ### Pseudocode
///
/// ```ignore
/// mask = [0; dims]
///
/// for i in range(dims):
///     mask[i] = a[i] < b[i] ? 1 : 0
///
/// return mask
/// ```
///
/// ### Note on `NaN` handling on `f32/f64` types
///
/// For `f32` and `f64` types, `NaN` values are handled as always being `false` in **ANY** comparison.
/// Even when compared against each other.
///
/// - `0.0 < 1.0 -> true`
/// - `0.0 < NaN -> false`
/// - `NaN < NaN -> false`
///
/// ### Result buffer
///
/// The result buffer can be either an initialized slice i.e. `&mut [T]`
/// or it can be a slice holding potentially uninitialized data i.e. `&mut [MaybeUninit<T>]`.
///
/// Once the operation is complete, it is safe to assume the data written is fully initialized.
///
/// ### Panics
///
/// Panics if the size of vectors `a`, `b` and `result` do not match.
pub fn lt_vertical<T, B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
where
    T: CmpOps,
    B1: IntoMemLoader<T>,
    B1::Loader: MemLoader<Value = T>,
    B2: IntoMemLoader<T>,
    B2::Loader: MemLoader<Value = T>,
    for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = T>,
{
    T::lt_vertical(lhs, rhs, result)
}

#[inline]
/// Checks each element pair from vectors `a` and `b` of size `dims`  comparing
/// if element `a` is **_less than or equal to_** element `b` returning a mask vector of the same type.
///
/// ### Pseudocode
///
/// ```ignore
/// mask = [0; dims]
///
/// for i in range(dims):
///     mask[i] = a[i] <= b[i] ? 1 : 0
///
/// return mask
/// ```
///
/// ### Note on `NaN` handling on `f32/f64` types
///
/// For `f32` and `f64` types, `NaN` values are handled as always being `false` in **ANY** comparison.
/// Even when compared against each other.
///
/// - `0.0 <= 1.0 -> true`
/// - `0.0 <= NaN -> false`
/// - `NaN <= NaN -> false`
///
/// ### Result buffer
///
/// The result buffer can be either an initialized slice i.e. `&mut [T]`
/// or it can be a slice holding potentially uninitialized data i.e. `&mut [MaybeUninit<T>]`.
///
/// Once the operation is complete, it is safe to assume the data written is fully initialized.
///
/// ### Panics
///
/// Panics if the size of vectors `a`, `b` and `result` do not match.
pub fn lte_vertical<T, B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
where
    T: CmpOps,
    B1: IntoMemLoader<T>,
    B1::Loader: MemLoader<Value = T>,
    B2: IntoMemLoader<T>,
    B2::Loader: MemLoader<Value = T>,
    for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = T>,
{
    T::lte_vertical(lhs, rhs, result)
}

#[inline]
/// Checks each element pair from vectors `a` and `b` of size `dims`  comparing
/// if element `a` is **_greater than_** element `b` returning a mask vector of the same type.
///
/// ### Pseudocode
///
/// ```ignore
/// mask = [0; dims]
///
/// for i in range(dims):
///     mask[i] = a[i] > b[i] ? 1 : 0
///
/// return mask
/// ```
///
/// ### Note on `NaN` handling on `f32/f64` types
///
/// For `f32` and `f64` types, `NaN` values are handled as always being `false` in **ANY** comparison.
/// Even when compared against each other.
///
/// - `1.0 > 0.0 -> true`
/// - `1.0 > NaN -> false`
/// - `NaN > NaN -> false`
///
/// ### Result buffer
///
/// The result buffer can be either an initialized slice i.e. `&mut [T]`
/// or it can be a slice holding potentially uninitialized data i.e. `&mut [MaybeUninit<T>]`.
///
/// Once the operation is complete, it is safe to assume the data written is fully initialized.
///
/// ### Panics
///
/// Panics if the size of vectors `a`, `b` and `result` do not match.
pub fn gt_vertical<T, B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
where
    T: CmpOps,
    B1: IntoMemLoader<T>,
    B1::Loader: MemLoader<Value = T>,
    B2: IntoMemLoader<T>,
    B2::Loader: MemLoader<Value = T>,
    for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = T>,
{
    T::gt_vertical(lhs, rhs, result)
}

#[inline]
/// Checks each element pair from vectors `a` and `b` of size `dims`  comparing
/// if element `a` is **_greater than_** element `b` returning a mask vector of the same type.
///
/// ### Pseudocode
///
/// ```ignore
/// mask = [0; dims]
///
/// for i in range(dims):
///     mask[i] = a[i] >= b[i] ? 1 : 0
///
/// return mask
/// ```
///
/// ### Note on `NaN` handling on `f32/f64` types
///
/// For `f32` and `f64` types, `NaN` values are handled as always being `false` in **ANY** comparison.
/// Even when compared against each other.
///
/// - `1.0 >= 0.0 -> true`
/// - `1.0 >= NaN -> false`
/// - `NaN >= NaN -> false`
///
/// ### Result buffer
///
/// The result buffer can be either an initialized slice i.e. `&mut [T]`
/// or it can be a slice holding potentially uninitialized data i.e. `&mut [MaybeUninit<T>]`.
///
/// Once the operation is complete, it is safe to assume the data written is fully initialized.
///
/// ### Panics
///
/// Panics if the size of vectors `a`, `b` and `result` do not match.
pub fn gte_vertical<T, B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
where
    T: CmpOps,
    B1: IntoMemLoader<T>,
    B1::Loader: MemLoader<Value = T>,
    B2: IntoMemLoader<T>,
    B2::Loader: MemLoader<Value = T>,
    for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = T>,
{
    T::gte_vertical(lhs, rhs, result)
}

/// Performs an element wise addition of each element pair of vector `a` and `b`,
/// writing the result to `result`.
///
/// ### Pseudocode
///
/// ```ignore
/// result = [0; dims]
///
/// for i in range(dims):
///     result[i] = a[i] + b[i]
///
/// return result
/// ```
///
/// ### Result buffer
///
/// The result buffer can be either an initialized slice i.e. `&mut [T]`
/// or it can be a slice holding potentially uninitialized data i.e. `&mut [MaybeUninit<T>]`.
///
/// Once the operation is complete, it is safe to assume the data written is fully initialized.
///
/// ### Panics
///
/// Panics if the size of vectors `a`, `b` and `result` do not match.
pub fn add_vertical<T, B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
where
    T: ArithmeticOps,
    B1: IntoMemLoader<T>,
    B1::Loader: MemLoader<Value = T>,
    B2: IntoMemLoader<T>,
    B2::Loader: MemLoader<Value = T>,
    for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = T>,
{
    T::add_vertical(lhs, rhs, result)
}

/// Performs an element wise subtraction of each element pair from vectors `a` and `b`,
/// writing the result to `result`.
///
/// ### Pseudocode
///
/// ```ignore
/// result = [0; dims]
///
/// for i in range(dims):
///     result[i] = a[i] - b[i]
///
/// return result
/// ```
///
/// ### Result buffer
///
/// The result buffer can be either an initialized slice i.e. `&mut [T]`
/// or it can be a slice holding potentially uninitialized data i.e. `&mut [MaybeUninit<T>]`.
///
/// Once the operation is complete, it is safe to assume the data written is fully initialized.
///
/// ### Panics
///
/// Panics if the size of vectors `a`, `b` and `result` do not match.
pub fn sub_vertical<T, B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
where
    T: ArithmeticOps,
    B1: IntoMemLoader<T>,
    B1::Loader: MemLoader<Value = T>,
    B2: IntoMemLoader<T>,
    B2::Loader: MemLoader<Value = T>,
    for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = T>,
{
    T::sub_vertical(lhs, rhs, result)
}

/// Performs an element wise multiplication of each element pair from vectors `a` and `b`,
/// writing the result to `result`.
///
/// ### Pseudocode
///
/// ```ignore
/// result = [0; dims]
///
/// for i in range(dims):
///     result[i] = a[i] * b[i]
///
/// return result
/// ```
///
/// ### Result buffer
///
/// The result buffer can be either an initialized slice i.e. `&mut [T]`
/// or it can be a slice holding potentially uninitialized data i.e. `&mut [MaybeUninit<T>]`.
///
/// Once the operation is complete, it is safe to assume the data written is fully initialized.
///
/// ### Panics
///
/// Panics if the size of vectors `a`, `b` and `result` do not match.
pub fn mul_vertical<T, B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
where
    T: ArithmeticOps,
    B1: IntoMemLoader<T>,
    B1::Loader: MemLoader<Value = T>,
    B2: IntoMemLoader<T>,
    B2::Loader: MemLoader<Value = T>,
    for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = T>,
{
    T::mul_vertical(lhs, rhs, result)
}

/// Performs an element wise division on each element pair from vectors `a` and `b`,
/// writing the result to `result`.
///
/// ### Pseudocode
///
/// ```ignore
/// result = [0; dims]
///
/// for i in range(dims):
///     result[i] = a[i] / b[i]
///
/// return result
/// ```
///
/// ### Result buffer
///
/// The result buffer can be either an initialized slice i.e. `&mut [T]`
/// or it can be a slice holding potentially uninitialized data i.e. `&mut [MaybeUninit<T>]`.
///
/// Once the operation is complete, it is safe to assume the data written is fully initialized.
///
/// ### Panics
///
/// Panics if the size of vectors `a`, `b` and `result` do not match.
pub fn div_vertical<T, B1, B2, B3>(lhs: B1, rhs: B2, result: &mut [B3])
where
    T: ArithmeticOps,
    B1: IntoMemLoader<T>,
    B1::Loader: MemLoader<Value = T>,
    B2: IntoMemLoader<T>,
    B2::Loader: MemLoader<Value = T>,
    for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = T>,
{
    T::div_vertical(lhs, rhs, result)
}
