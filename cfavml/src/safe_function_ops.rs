//! All CFAVML routines exposed as generic functions.
//!
//! These functions use runtime dispatch when the `std` feature is enabled, otherwise in
//! no std environments, compile time dispatch is used by detecting the `target_feature` cfg.
//!
//! This means you when runtime detection is disabled, you must compile with one of `target-cpu`
//! or `target-feature` Rust flags set otherwise this will always use the `Fallback` implementations.

use crate::buffer::WriteOnlyBuffer;
use crate::safe_trait_agg_ops::AggOps;
use crate::safe_trait_arithmetic_ops::ArithmeticOps;
use crate::safe_trait_cmp_ops::CmpOps;
use crate::safe_trait_distance_ops::DistanceOps;

#[inline]
/// Calculates the cosine similarity distance of vectors `a` and `b`.
///
/// ### Pseudocode
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
pub fn cosine<T: DistanceOps>(a: &[T], b: &[T]) -> T {
    T::cosine(a.len(), a, b)
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
pub fn dot<T: DistanceOps>(a: &[T], b: &[T]) -> T {
    T::dot(a.len(), a, b)
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
pub fn squared_euclidean<T: DistanceOps>(a: &[T], b: &[T]) -> T {
    T::squared_euclidean(a.len(), a, b)
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
pub fn squared_norm<T: DistanceOps>(a: &[T]) -> T {
    T::squared_norm(a.len(), a)
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
pub fn sum<T: AggOps>(a: &[T]) -> T {
    T::sum(a.len(), a)
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
pub fn max_horizontal<T: CmpOps>(a: &[T]) -> T {
    T::max_horizontal(a.len(), a)
}

#[inline]
/// Performs an element wise max on each element of vector `a` and the provided broadcast
/// value, writing the result to `result`.
///
/// ### Pseudocode
///
/// ```ignore
/// result = [0; dims]
///
/// for i in range(dims):
///     result[i] = max(value, a[i])
///
/// return result
/// ```
///
/// ### Result buffer
///
/// The result buffer can be either an initialized slice i.e. [`&mut [Self]`]
/// or it can be a slice holding potentially uninitialized data i.e. [`&mut [MaybeUninit<Self>]`].
///
/// Once the operation is complete, it is safe to assume the data written is fully initialized.
///
/// ### Panics
///
/// Panics if the size of vector `a` and `result` do not match.
pub fn max_value<T, B>(value: T, a: &[T], result: &mut [B])
where
    T: CmpOps,
    for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
{
    T::max_value(a.len(), value, a, result)
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
/// The result buffer can be either an initialized slice i.e. [`&mut [Self]`]
/// or it can be a slice holding potentially uninitialized data i.e. [`&mut [MaybeUninit<Self>]`].
///
/// Once the operation is complete, it is safe to assume the data written is fully initialized.
///
/// ### Panics
///
/// Panics if the size of vectors `a`, `b` and `result` do not match.
pub fn max_vector<T, B>(a: &[T], b: &[T], result: &mut [B])
where
    T: CmpOps,
    for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
{
    T::max_vector(a.len(), a, b, result)
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
pub fn min_horizontal<T: CmpOps>(a: &[T]) -> T {
    T::min_horizontal(a.len(), a)
}

#[inline]
/// Performs an element wise min on each element of vector `a` and the provided broadcast
/// value, writing the result to `result`.
///
/// ### Pseudocode
///
/// ```ignore
/// result = [0; dims]
///
/// for i in range(dims):
///     result[i] = min(value, a[i])
///
/// return result
/// ```
///
/// ### Result buffer
///
/// The result buffer can be either an initialized slice i.e. [`&mut [Self]`]
/// or it can be a slice holding potentially uninitialized data i.e. [`&mut [MaybeUninit<Self>]`].
///
/// Once the operation is complete, it is safe to assume the data written is fully initialized.
///
/// ### Panics
///
/// Panics if the size of vectors `a` and `result` do not match.
pub fn min_value<T, B>(value: T, a: &[T], result: &mut [B])
where
    T: CmpOps,
    for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
{
    T::min_value(a.len(), value, a, result)
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
/// The result buffer can be either an initialized slice i.e. [`&mut [Self]`]
/// or it can be a slice holding potentially uninitialized data i.e. [`&mut [MaybeUninit<Self>]`].
///
/// Once the operation is complete, it is safe to assume the data written is fully initialized.
///
/// ### Panics
///
/// Panics if the size of vectors `a`, `b` and `result` do not match.
pub fn min_vector<T, B>(a: &[T], b: &[T], result: &mut [B])
where
    T: CmpOps,
    for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
{
    T::min_vector(a.len(), a, b, result)
}

/// Performs an element wise addition of each element of vector `a` and the provided broadcast
/// value, writing the result to `result`.
///
/// ### Pseudocode
///
/// ```ignore
/// result = [0; dims]
///
/// for i in range(dims):
///     result[i] = a[i] + value
///
/// return result
/// ```
///
/// ### Result buffer
///
/// The result buffer can be either an initialized slice i.e. [`&mut [Self]`]
/// or it can be a slice holding potentially uninitialized data i.e. [`&mut [MaybeUninit<Self>]`].
///
/// Once the operation is complete, it is safe to assume the data written is fully initialized.
///
/// ### Panics
///
/// Panics if the size of vectors `a` and `result` do not match.
pub fn add_value<T, B>(value: T, a: &[T], result: &mut [B])
where
    T: ArithmeticOps,
    for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
{
    T::add_value(a.len(), value, a, result)
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
/// The result buffer can be either an initialized slice i.e. [`&mut [Self]`]
/// or it can be a slice holding potentially uninitialized data i.e. [`&mut [MaybeUninit<Self>]`].
///
/// Once the operation is complete, it is safe to assume the data written is fully initialized.
///
/// ### Panics
///
/// Panics if the size of vectors `a`, `b` and `result` do not match.
pub fn add_vector<T, B>(a: &[T], b: &[T], result: &mut [B])
where
    T: ArithmeticOps,
    for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
{
    T::add_vector(a.len(), a, b, result)
}

/// Performs an element wise subtraction of each element of vector `a` and the provided broadcast
/// value, writing the result to `result`.
///
/// ### Pseudocode
///
/// ```ignore
/// result = [0; dims]
///
/// for i in range(dims):
///     result[i] = a[i] - value
///
/// return result
/// ```
///
/// ### Result buffer
///
/// The result buffer can be either an initialized slice i.e. [`&mut [Self]`]
/// or it can be a slice holding potentially uninitialized data i.e. [`&mut [MaybeUninit<Self>]`].
///
/// Once the operation is complete, it is safe to assume the data written is fully initialized.
///
/// ### Panics
///
/// Panics if the size of vectors `a` and `result` do not match.
pub fn sub_value<T, B>(value: T, a: &[T], result: &mut [B])
where
    T: ArithmeticOps,
    for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
{
    T::sub_value(a.len(), value, a, result)
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
/// The result buffer can be either an initialized slice i.e. [`&mut [Self]`]
/// or it can be a slice holding potentially uninitialized data i.e. [`&mut [MaybeUninit<Self>]`].
///
/// Once the operation is complete, it is safe to assume the data written is fully initialized.
///
/// ### Panics
///
/// Panics if the size of vectors `a`, `b` and `result` do not match.
pub fn sub_vector<T, B>(a: &[T], b: &[T], result: &mut [B])
where
    T: ArithmeticOps,
    for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
{
    T::sub_vector(a.len(), a, b, result)
}

/// Performs an element wise multiplication of each element of vector `a` and the provided broadcast
/// value, writing the result to `result`.
///
/// ### Pseudocode
///
/// ```ignore
/// result = [0; dims]
///
/// for i in range(dims):
///     result[i] = a[i] * value
///
/// return result
/// ```
///
/// ### Result buffer
///
/// The result buffer can be either an initialized slice i.e. [`&mut [Self]`]
/// or it can be a slice holding potentially uninitialized data i.e. [`&mut [MaybeUninit<Self>]`].
///
/// Once the operation is complete, it is safe to assume the data written is fully initialized.
///
/// ### Panics
///
/// Panics if the size of vectors `a` and `result` do not match.
pub fn mul_value<T, B>(value: T, a: &[T], result: &mut [B])
where
    T: ArithmeticOps,
    for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
{
    T::mul_value(a.len(), value, a, result)
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
/// The result buffer can be either an initialized slice i.e. [`&mut [Self]`]
/// or it can be a slice holding potentially uninitialized data i.e. [`&mut [MaybeUninit<Self>]`].
///
/// Once the operation is complete, it is safe to assume the data written is fully initialized.
///
/// ### Panics
///
/// Panics if the size of vectors `a`, `b` and `result` do not match.
pub fn mul_vector<T, B>(a: &[T], b: &[T], result: &mut [B])
where
    T: ArithmeticOps,
    for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
{
    T::mul_vector(a.len(), a, b, result)
}

/// Performs an element wise division of each element of vector `a` and the provided broadcast
/// value, writing the result to `result`.
///
/// ### Pseudocode
///
/// ```ignore
/// result = [0; dims]
///
/// for i in range(dims):
///     result[i] = a[i] / value
///
/// return result
/// ```
///
/// ### Result buffer
///
/// The result buffer can be either an initialized slice i.e. [`&mut [Self]`]
/// or it can be a slice holding potentially uninitialized data i.e. [`&mut [MaybeUninit<Self>]`].
///
/// Once the operation is complete, it is safe to assume the data written is fully initialized.
///
/// ### Panics
///
/// Panics if the size of vectors `a` and `result` do not match.
pub fn div_value<T, B>(value: T, a: &[T], result: &mut [B])
where
    T: ArithmeticOps,
    for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
{
    T::div_value(a.len(), value, a, result)
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
/// The result buffer can be either an initialized slice i.e. [`&mut [Self]`]
/// or it can be a slice holding potentially uninitialized data i.e. [`&mut [MaybeUninit<Self>]`].
///
/// Once the operation is complete, it is safe to assume the data written is fully initialized.
///
/// ### Panics
///
/// Panics if the size of vectors `a`, `b` and `result` do not match.
pub fn div_vector<T, B>(a: &[T], b: &[T], result: &mut [B])
where
    T: ArithmeticOps,
    for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
{
    T::div_vector(a.len(), a, b, result)
}
