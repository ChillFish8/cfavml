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
/// ### Examples
///
/// We can create two vectors and calculate the cosine distance _providing they are the same length_.
/// Any type that implements `AsRef<[A]>` can be provided, where `A` is any type from:
///
/// > `f32`, `f64`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`
///
/// _Although you likely want `f32` or `f64`._
///
/// ```rust
/// let a = vec![1.0, 0.3, 0.2, 0.4, 0.2, 0.1, 0.3, 0.2];
/// let b = vec![0.8, 0.2, 0.1, 0.4, 0.2, 0.5, 0.8, 0.4];
///
/// let distance = cfavml::cosine(&a, &b);
/// assert_eq!(distance, 0.14136523227140463);
/// ```
///
/// ### Implementation Pseudocode
///
/// _This is the logic of the routine being called._
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
    B1::Loader: MemLoader<Value = T>,
    B2: IntoMemLoader<T>,
    B2::Loader: MemLoader<Value = T>,
{
    T::cosine(a, b)
}

#[inline]
/// Calculates the cosine similarity distance of vectors `a` and `b`.
///
/// ### Examples
///
/// We can create two vectors and calculate the dot product _providing they are the same length_.
/// Any type that implements `AsRef<[A]>` can be provided, where `A` is any type from:
///
/// > `f32`, `f64`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`
///
/// _Although you likely want `f32` or `f64`._
///
/// ```rust
/// let a = vec![1.0, 0.3, 0.2, 0.4, 0.2, 0.4, 0.3, 0.2];
/// let b = vec![0.8, 0.2, 0.1, 0.4, 0.2, 0.4, 0.8, 0.4];
///
/// let distance = cfavml::dot(&a, &b);
/// assert_eq!(distance, 1.56);
/// ```
///
/// ### Implementation Pseudocode
///
/// _This is the logic of the routine being called._
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
    B1::Loader: MemLoader<Value = T>,
    B2: IntoMemLoader<T>,
    B2::Loader: MemLoader<Value = T>,
{
    T::dot(a, b)
}

#[inline]
/// Calculates the squared Euclidean distance of vectors `a` and `b`.
///
/// ### Examples
///
/// We can create two vectors and calculate the squared Euclidean distance _providing they are the same length_.
/// Any type that implements `AsRef<[A]>` can be provided, where `A` is any type from:
///
/// > `f32`, `f64`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`
///
/// _Although you likely want `f32` or `f64`._
///
/// ```rust
/// let a = vec![1.0, 0.3, 0.2, 0.4, 0.2, 0.1, 0.3, 0.2];
/// let b = vec![0.8, 0.2, 0.1, 0.4, 0.2, 0.5, 0.8, 0.4];
///
/// let distance = cfavml::squared_euclidean(&a, &b);
/// assert_eq!(distance, 0.51);
/// ```
///
/// ### Implementation Pseudocode
///
/// _This is the logic of the routine being called._
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
    B1::Loader: MemLoader<Value = T>,
    B2: IntoMemLoader<T>,
    B2::Loader: MemLoader<Value = T>,
{
    T::squared_euclidean(a, b)
}

#[inline]
/// Calculates the squared L2 norm of vector `a`.
///
/// ### Examples
///
/// We can create a single vector and calculate the squared L2 norm.
/// Any type that implements `AsRef<[A]>` can be provided, where `A` is any type from:
///
/// > `f32`, `f64`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`
///
/// _Although you likely want `f32` or `f64`._
///
/// ```rust
/// let a = vec![1.0, 0.3, 0.2, 0.4, 0.2, 0.1, 0.3, 0.2];
///
/// let norm = cfavml::squared_norm(&a);
/// assert_eq!(norm, 1.47);
/// ```
///
/// ### Implementation Pseudocode
///
/// _This is the logic of the routine being called._
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
    B1::Loader: MemLoader<Value = T>,
{
    T::squared_norm(a)
}

#[inline]
/// Performs a horizontal sum of all elements in a returning the result.
///
/// ### Examples
///
/// We can create a single vector and calculate the squared L2 norm.
/// Any type that implements `AsRef<[A]>` can be provided, where `A` is any type from:
///
/// > `f32`, `f64`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`
///
/// It is worth noting however, the compiler can often match the speed of this particular
/// routine if your operations are as simple as `my_vector.iter().sum()`.
///
/// ```rust
/// let a = vec![1.0, 0.3, 0.2, 0.4, 0.2, 0.1, 0.3, 0.2];
///
/// let total = cfavml::sum(&a);
/// assert_eq!(total, 2.7);
/// ```
///
/// ### Implementation Pseudocode
///
/// _This is the logic of the routine being called._
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
    B1::Loader: MemLoader<Value = T>,
{
    T::sum(a)
}

#[inline]
/// Finds the horizontal max element of a given vector and returns the result.
///
/// ### Default Value Warning
///
/// Beware of the default value returned when passing a zero-length array (which is technically allowed)
///
/// If you are using `f32` or `f64` types, this becomes `T::NEG_INFINITY` otherwise the
/// default max value is `T:MIN`.
///
/// ### Examples
///
/// Any vector can be created and have the maximum value found within using the `max` operation, but
/// be aware of the default value handling if you pass an empty array
/// (see the "Default Value Warning" section for more.)
///
/// ```rust
/// let a = vec![1.0, 0.3, 0.2, 0.4, 0.2, 0.1, 0.3, 0.2];
///
/// let total = cfavml::max(&a);
/// assert_eq!(total, 1.0);
/// ```
///
/// ### Implementation Pseudocode
///
/// _This is the logic of the routine being called._
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
    B1::Loader: MemLoader<Value = T>,
{
    T::max(a)
}

#[inline]
/// Takes the element wise max of vectors `a` and `b` of size `dims` and stores the result
/// in `result` of size `dims`.
///
/// ### Things To Know
///
/// ###### Supported patterns
///
/// Unlike horizontal operations, vertical ops can execute over a much wider variety of data
/// depending on the `MemLoader` which is a trait used to control how inputs are projected and
/// read from before executing. For the most part, you don't need to worry about this outside of
/// knowing you can by default, pass one combination of:
///     
/// - `lhs: vector` and `rhs: vector`  
/// - `lhs: vector` and `rhs: broadcast value`  
/// - `lhs: broadcast value` and `rhs: vector`  
/// - `lhs: broadcast value` and `rhs: broadcast value`  
///   * Not really that useful and is just an artefact of the memory management system.
///
/// ###### Broadcast values
///
/// When a broadcast value is provided, CFAVML will stretch that value out to match the size of
/// the _result_ buffer (not the other input buffer!) this does not cost additional allocations
/// outside the result buffer itself.
///
/// This means the following is possible:
/// - `[0, 0, 0] + 1 == [1, 1, 1]`  w/result_buffer_len=3
/// - `0 + 1 == [1, 1, 1]`  w/result_buffer_len=3
/// - `1 + 1 == [1]`  w/result_buffer_len=1
///
/// ### Examples
///
/// ##### Two vectors
///
/// ```rust
/// let lhs = [1.0, 1.0, 1.0, 1.0];
/// let rhs = [2.0, 2.5, 1.0, -2.0];
///
/// let mut result = [0.0; 4];
/// cfavml::max_vertical(&lhs, &rhs, &mut result);
/// assert_eq!(result, [2.0, 2.5, 1.0, 1.0]);
/// ```
///
/// ##### One vector & broadcast value
///
/// ```rust
/// let lhs = [2.0, 3.0, -1.0, -0.5];
///
/// let mut result = [0.0; 4];
/// cfavml::max_vertical(&lhs, 0.0, &mut result);
/// assert_eq!(result, [2.0, 3.0, 0.0, 0.0]);
/// ```
///
/// ##### Two broadcast values
///
/// ```rust
/// let mut result = [0.0; 4];
/// cfavml::max_vertical(1.0, 0.0, &mut result);
/// assert_eq!(result, [1.0; 4]);
/// ```
///
/// ##### With `MaybeUninit`
///
/// Often if you are working with new-allocations, you do not want to initialize the data twice,
/// CFAVML guarantees that the output buffer will never be read from, so it is safe to provide
/// uninitialized buffers for the result, this is what the `WriteOnlyBuffer` trait is about.
///
/// ```rust
/// use core::mem::MaybeUninit;
///
/// let lhs = [1.0, 1.0, 1.0, 1.0];
/// let rhs = [2.0, 2.5, 1.0, -2.0];
///
/// let mut result = Vec::with_capacity(4);
/// unsafe { result.set_len(4) };
/// cfavml::max_vertical(&lhs, &rhs, &mut result);
///
/// let result = unsafe { core::mem::transmute::<Vec<MaybeUninit<f32>>, Vec<f32>>(result) };
/// assert_eq!(result, [2.0, 2.5, 1.0, 1.0]);
/// ```
///
/// ### Projecting Vectors
///
/// CFAVML allows for working over a wide variety of buffers for applications, projection is effectively
/// broadcasting of two input buffers implementing `IntoMemLoader<T>`.
///
/// By default, you can provide _two slices_, _one slice and a broadcast value_, or _two broadcast values_,
/// which exhibit the standard behaviour as you might expect.
///
/// When providing two slices as inputs they cannot be projected to a buffer
/// that is larger their input sizes by default. This means providing two slices
/// of `128` elements in length must take a result buffer of `128` elements in length.
///
/// ### Implementation Pseudocode
///
/// _This is the logic of the routine being called._
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
/// # Panics
///
/// If vectors `a` and `b` cannot be projected to the target size of `result`.
/// Note that the projection rules are tied to the `MemLoader` implementation.
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
/// ### Default Value Warning
///
/// Beware of the default value returned when passing a zero-length array (which is technically allowed)
///
/// If you are using `f32` or `f64` types, this becomes `T::NEG_INFINITY` otherwise the
/// default max value is `T:MIN`.
///
/// ### Examples
///
/// Any vector can be created and have the minimum value found within using the `min` operation, but
/// be aware of the default value handling if you pass an empty array
/// (see the "Default Value Warning" section for more.)
///
/// ```rust
/// let a = vec![1.0, 0.3, 0.2, 0.4, 0.2, 0.1, 0.3, 0.2];
///
/// let total = cfavml::min(&a);
/// assert_eq!(total, 0.1);
/// ```
///
/// ### Implementation Pseudocode
///
/// _This is the logic of the routine being called._
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
    B1::Loader: MemLoader<Value = T>,
{
    T::min(a)
}

#[inline]
/// Takes the element wise min of vectors `a` and `b` of size `dims` and stores the result
/// in `result` of size `dims`.
///
/// ### Things To Know
///
/// ###### Supported patterns
///
/// Unlike horizontal operations, vertical ops can execute over a much wider variety of data
/// depending on the `MemLoader` which is a trait used to control how inputs are projected and
/// read from before executing. For the most part, you don't need to worry about this outside of
/// knowing you can by default, pass one combination of:
///     
/// - `lhs: vector` and `rhs: vector`  
/// - `lhs: vector` and `rhs: broadcast value`  
/// - `lhs: broadcast value` and `rhs: vector`  
/// - `lhs: broadcast value` and `rhs: broadcast value`  
///   * Not really that useful and is just an artefact of the memory management system.
///
/// ###### Broadcast values
///
/// When a broadcast value is provided, CFAVML will stretch that value out to match the size of
/// the _result_ buffer (not the other input buffer!) this does not cost additional allocations
/// outside the result buffer itself.
///
/// This means the following is possible:
/// - `[0, 0, 0] + 1 == [1, 1, 1]`  w/result_buffer_len=3
/// - `0 + 1 == [1, 1, 1]`  w/result_buffer_len=3
/// - `1 + 1 == [1]`  w/result_buffer_len=1
///
/// ### Examples
///
/// ##### Two vectors
///
/// ```rust
/// let lhs = [1.0, 1.0, 1.0, 1.0];
/// let rhs = [2.0, 2.5, 1.0, -2.0];
///
/// let mut result = [0.0; 4];
/// cfavml::min_vertical(&lhs, &rhs, &mut result);
/// assert_eq!(result, [1.0, 1.0, 1.0, -2.0]);
/// ```
///
/// ##### One vector & broadcast value
///
/// ```rust
/// let lhs = [2.0, 3.0, -1.0, -0.5];
///
/// let mut result = [0.0; 4];
/// cfavml::min_vertical(&lhs, 0.0, &mut result);
/// assert_eq!(result, [0.0, 0.0, -1.0, -0.5]);
/// ```
///
/// ##### Two broadcast values
///
/// ```rust
/// let mut result = [0.0; 4];
/// cfavml::min_vertical(1.0, -5.0, &mut result);
/// assert_eq!(result, [-5.0; 4]);
/// ```
///
/// ##### With `MaybeUninit`
///
/// Often if you are working with new-allocations, you do not want to initialize the data twice,
/// CFAVML guarantees that the output buffer will never be read from, so it is safe to provide
/// uninitialized buffers for the result, this is what the `WriteOnlyBuffer` trait is about.
///
/// ```rust
/// use core::mem::MaybeUninit;
///
/// let lhs = [1.0, -1.0, 0.5, 1.0];
/// let rhs = [2.0, 2.5, 1.0, -2.0];
///
/// let mut result = Vec::with_capacity(4);
/// unsafe { result.set_len(4) };
/// cfavml::min_vertical(&lhs, &rhs, &mut result);
///
/// let result = unsafe { core::mem::transmute::<Vec<MaybeUninit<f32>>, Vec<f32>>(result) };
/// assert_eq!(result, [1.0, -1.0, 0.5, -2.0]);
/// ```
///
/// ### Projecting Vectors
///
/// CFAVML allows for working over a wide variety of buffers for applications, projection is effectively
/// broadcasting of two input buffers implementing `IntoMemLoader<T>`.
///
/// By default, you can provide _two slices_, _one slice and a broadcast value_, or _two broadcast values_,
/// which exhibit the standard behaviour as you might expect.
///
/// When providing two slices as inputs they cannot be projected to a buffer
/// that is larger their input sizes by default. This means providing two slices
/// of `128` elements in length must take a result buffer of `128` elements in length.
///
/// ### Implementation Pseudocode
///
/// _This is the logic of the routine being called._
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
/// # Panics
///
/// If vectors `a` and `b` cannot be projected to the target size of `result`.
/// Note that the projection rules are tied to the `MemLoader` implementation.
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
/// Checks each element pair of elements from vectors `a` and `b` comparing if
/// element `a` is **_equal to_** element `b`, storing the output as `1` (true) or `0` (false)
/// in `result`.
///
/// ### Things To Know
///
/// ###### Supported patterns
///
/// Unlike horizontal operations, vertical ops can execute over a much wider variety of data
/// depending on the `MemLoader` which is a trait used to control how inputs are projected and
/// read from before executing. For the most part, you don't need to worry about this outside of
/// knowing you can by default, pass one combination of:
///     
/// - `lhs: vector` and `rhs: vector`  
/// - `lhs: vector` and `rhs: broadcast value`  
/// - `lhs: broadcast value` and `rhs: vector`  
/// - `lhs: broadcast value` and `rhs: broadcast value`  
///   * Not really that useful and is just an artefact of the memory management system.
///
/// ###### Broadcast values
///
/// When a broadcast value is provided, CFAVML will stretch that value out to match the size of
/// the _result_ buffer (not the other input buffer!) this does not cost additional allocations
/// outside the result buffer itself.
///
/// This means the following is possible:
/// - `[0, 0, 0] + 1 == [1, 1, 1]`  w/result_buffer_len=3
/// - `0 + 1 == [1, 1, 1]`  w/result_buffer_len=3
/// - `1 + 1 == [1]`  w/result_buffer_len=1
///
/// ###### Masks
///
/// CFAVML follows the same pattern as numpy, which is it representing boolean results as
/// either `1` or `0` in the respective type. This allows you to do various bit manipulation
/// and arithmetic techniques for processing values within the vector.
///
/// ### Examples
///
/// ##### Two vectors
///
/// ```rust
/// let lhs = [1.0, 2.3, 2.0, 1.0];
/// let rhs = [2.0, 0.7, 2.0, -2.0];
///
/// let mut mask = [0.0f32; 4];
/// cfavml::eq_vertical(&lhs, &rhs, &mut mask);
/// assert_eq!(mask, [0.0, 0.0, 1.0, 0.0]);   // Value at index 2 as equal!
///
/// // Now we can use it to zero any values that don't match.
/// let mut match_or_zeroes = [0.0f32; 4];
/// cfavml::mul_vertical(&lhs, &mask, &mut match_or_zeroes);
///
/// // Our original match is extracted and the rest are `0.0`
/// // For convenience, I've used `0.0` as the non-match value,
/// // but if you switch `mask` and `lhs` around you can get
/// // a `NaN` mask which may be more useful depending on application.
/// assert_eq!(match_or_zeroes, [0.0, 0.0, 2.0, 0.0]);    
/// ```
///
/// ##### One vector & broadcast value
///
/// ```rust
/// let lhs = [2.0, f32::NAN, -1.0, -0.5];
///
/// let mut result = [0.0f32; 4];
/// cfavml::eq_vertical(&lhs, -0.5, &mut result);
/// assert_eq!(result, [0.0, 0.0, 0.0, 1.0]);  // NaN is always false on eq checks.
/// ```
///
/// ##### Two broadcast values
///
/// ```rust
/// let mut result = [0.0f32; 4];
/// cfavml::eq_vertical(1.0, -5.0, &mut result);
/// assert_eq!(result, [0.0; 4]);
/// ```
///
/// ##### With `MaybeUninit`
///
/// Often if you are working with new-allocations, you do not want to initialize the data twice,
/// CFAVML guarantees that the output buffer will never be read from, so it is safe to provide
/// uninitialized buffers for the result, this is what the `WriteOnlyBuffer` trait is about.
///
/// ```rust
/// use core::mem::MaybeUninit;
///
/// let lhs = [1.0, -1.0, 0.5, 1.0];
/// let rhs = [1.0, 2.5, 0.5, -2.0];
///
/// let mut result = Vec::with_capacity(4);
/// unsafe { result.set_len(4) };
/// cfavml::eq_vertical(&lhs, &rhs, &mut result);
///
/// let result = unsafe { core::mem::transmute::<Vec<MaybeUninit<f32>>, Vec<f32>>(result) };
/// assert_eq!(result, [1.0, 0.0, 1.0, 0.0]);
/// ```
///
/// ### Projecting Vectors
///
/// CFAVML allows for working over a wide variety of buffers for applications, projection is effectively
/// broadcasting of two input buffers implementing `IntoMemLoader<T>`.
///
/// By default, you can provide _two slices_, _one slice and a broadcast value_, or _two broadcast values_,
/// which exhibit the standard behaviour as you might expect.
///
/// When providing two slices as inputs they cannot be projected to a buffer
/// that is larger their input sizes by default. This means providing two slices
/// of `128` elements in length must take a result buffer of `128` elements in length.
///
/// ### Implementation Pseudocode
///
/// _This is the logic of the routine being called._
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
/// # Panics
///
/// If vectors `a` and `b` cannot be projected to the target size of `result`.
/// Note that the projection rules are tied to the `MemLoader` implementation.
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
/// Checks each element pair of elements from vectors `a` and `b` comparing if
/// element `a` is **_not equal to_** element `b`, storing the output as `1` (true)
/// or `0` (false) in `result`.
///
/// ### Things To Know
///
/// ###### Supported patterns
///
/// Unlike horizontal operations, vertical ops can execute over a much wider variety of data
/// depending on the `MemLoader` which is a trait used to control how inputs are projected and
/// read from before executing. For the most part, you don't need to worry about this outside of
/// knowing you can by default, pass one combination of:
///     
/// - `lhs: vector` and `rhs: vector`  
/// - `lhs: vector` and `rhs: broadcast value`  
/// - `lhs: broadcast value` and `rhs: vector`  
/// - `lhs: broadcast value` and `rhs: broadcast value`  
///   * Not really that useful and is just an artefact of the memory management system.
///
/// ###### Broadcast values
///
/// When a broadcast value is provided, CFAVML will stretch that value out to match the size of
/// the _result_ buffer (not the other input buffer!) this does not cost additional allocations
/// outside the result buffer itself.
///
/// This means the following is possible:
/// - `[0, 0, 0] + 1 == [1, 1, 1]`  w/result_buffer_len=3
/// - `0 + 1 == [1, 1, 1]`  w/result_buffer_len=3
/// - `1 + 1 == [1]`  w/result_buffer_len=1
///
/// ###### Masks
///
/// CFAVML follows the same pattern as numpy, which is it representing boolean results as
/// either `1` or `0` in the respective type. This allows you to do various bit manipulation
/// and arithmetic techniques for processing values within the vector.
///
/// ### Examples
///
/// ##### Two vectors
///
/// ```rust
/// let lhs = [1.0, 2.3, 2.0, 1.0];
/// let rhs = [2.0, 0.7, 2.0, -2.0];
///
/// let mut mask = [0.0f32; 4];
/// cfavml::neq_vertical(&lhs, &rhs, &mut mask);
/// assert_eq!(mask, [1.0, 1.0, 0.0, 1.0]);   // All values except index 2 are not equal!
///
/// // Now we can use it to zero any values that are equal
/// let mut match_or_zeroes = [0.0f32; 4];
/// cfavml::mul_vertical(&lhs, &mask, &mut match_or_zeroes);
///
/// // Our original match is extracted and the rest are `0.0`
/// // For convenience, I've used `0.0` as the non-match value,
/// // but if you switch `mask` and `lhs` around you can get
/// // a `NaN` mask which may be more useful depending on application.
/// assert_eq!(match_or_zeroes, [1.0, 2.3, 0.0, 1.0]);    
/// ```
///
/// ##### One vector & broadcast value
///
/// ```rust
/// let lhs = [2.0, f32::NAN, -1.0, -0.5];
///
/// let mut result = [0.0f32; 4];
/// cfavml::neq_vertical(&lhs, -0.5, &mut result);
/// assert_eq!(result, [1.0, 1.0, 1.0, 0.0]);  // NaN is always false on eq checks.
/// ```
///
/// ##### Two broadcast values
///
/// ```rust
/// let mut result = [0.0f32; 4];
/// cfavml::neq_vertical(1.0, -5.0, &mut result);
/// assert_eq!(result, [1.0; 4]);
/// ```
///
/// ##### With `MaybeUninit`
///
/// Often if you are working with new-allocations, you do not want to initialize the data twice,
/// CFAVML guarantees that the output buffer will never be read from, so it is safe to provide
/// uninitialized buffers for the result, this is what the `WriteOnlyBuffer` trait is about.
///
/// ```rust
/// use core::mem::MaybeUninit;
///
/// let lhs = [1.0, -1.0, 0.5, 1.0];
/// let rhs = [1.0, 2.5, 0.5, -2.0];
///
/// let mut result = Vec::with_capacity(4);
/// unsafe { result.set_len(4) };
/// cfavml::neq_vertical(&lhs, &rhs, &mut result);
///
/// let result = unsafe { core::mem::transmute::<Vec<MaybeUninit<f32>>, Vec<f32>>(result) };
/// assert_eq!(result, [0.0, 1.0, 0.0, 1.0]);
/// ```
///
/// ### Projecting Vectors
///
/// CFAVML allows for working over a wide variety of buffers for applications, projection is effectively
/// broadcasting of two input buffers implementing `IntoMemLoader<T>`.
///
/// By default, you can provide _two slices_, _one slice and a broadcast value_, or _two broadcast values_,
/// which exhibit the standard behaviour as you might expect.
///
/// When providing two slices as inputs they cannot be projected to a buffer
/// that is larger their input sizes by default. This means providing two slices
/// of `128` elements in length must take a result buffer of `128` elements in length.
///
/// ### Implementation Pseudocode
///
/// _This is the logic of the routine being called._
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
/// Even when compared against each other.
///
/// - `0.0 != 1.0 -> true`
/// - `0.0 != NaN -> true`
/// - `NaN != NaN -> true`
///
/// # Panics
///
/// If vectors `a` and `b` cannot be projected to the target size of `result`.
/// Note that the projection rules are tied to the `MemLoader` implementation.
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
/// Checks each element pair of elements from vectors `a` and `b` comparing if
/// element `a` is **_less than_** element `b`, storing the output as `1` (true)
/// or `0` (false) in `result`.
///
/// ### Things To Know
///
/// ###### Supported patterns
///
/// Unlike horizontal operations, vertical ops can execute over a much wider variety of data
/// depending on the `MemLoader` which is a trait used to control how inputs are projected and
/// read from before executing. For the most part, you don't need to worry about this outside of
/// knowing you can by default, pass one combination of:
///     
/// - `lhs: vector` and `rhs: vector`  
/// - `lhs: vector` and `rhs: broadcast value`  
/// - `lhs: broadcast value` and `rhs: vector`  
/// - `lhs: broadcast value` and `rhs: broadcast value`  
///   * Not really that useful and is just an artefact of the memory management system.
///
/// ###### Broadcast values
///
/// When a broadcast value is provided, CFAVML will stretch that value out to match the size of
/// the _result_ buffer (not the other input buffer!) this does not cost additional allocations
/// outside the result buffer itself.
///
/// This means the following is possible:
/// - `[0, 0, 0] + 1 == [1, 1, 1]`  w/result_buffer_len=3
/// - `0 + 1 == [1, 1, 1]`  w/result_buffer_len=3
/// - `1 + 1 == [1]`  w/result_buffer_len=1
///
/// ###### Masks
///
/// CFAVML follows the same pattern as numpy, which is it representing boolean results as
/// either `1` or `0` in the respective type. This allows you to do various bit manipulation
/// and arithmetic techniques for processing values within the vector.
///
/// ### Examples
///
/// ##### Two vectors
///
/// ```rust
/// let lhs = [-1.0, 2.3, 2.0, 1.0];
/// let rhs = [2.0, 0.7, 2.0, -2.0];
///
/// let mut mask = [0.0f32; 4];
/// cfavml::lt_vertical(&lhs, &rhs, &mut mask);
/// assert_eq!(mask, [1.0, 0.0, 0.0, 0.0]);  // Only index 0 of lhs is less than the rhs.
///
/// // Now we can use it to zero any values that greater than or equal.
/// let mut match_or_zeroes = [0.0f32; 4];
/// cfavml::mul_vertical(&lhs, &mask, &mut match_or_zeroes);
///
/// // Our original match is extracted and the rest are `0.0`
/// // For convenience, I've used `0.0` as the non-match value,
/// // but if you switch `mask` and `lhs` around you can get
/// // a `NaN` mask which may be more useful depending on application.
/// assert_eq!(match_or_zeroes, [-1.0, 0.0, 0.0, 0.0]);    
/// ```
///
/// ##### One vector & broadcast value
///
/// ```rust
/// let lhs = [2.0, f32::NAN, -1.0, -0.5];
///
/// let mut result = [0.0f32; 4];
/// cfavml::lt_vertical(&lhs, -0.5, &mut result);
/// assert_eq!(result, [0.0, 0.0, 1.0, 0.0]);  // NaN is always false on lt checks.
/// ```
///
/// ##### Two broadcast values
///
/// ```rust
/// let mut result = [0.0f32; 4];
/// cfavml::lt_vertical(-1.0, 5.0, &mut result);
/// assert_eq!(result, [1.0; 4]);
/// ```
///
/// ##### With `MaybeUninit`
///
/// Often if you are working with new-allocations, you do not want to initialize the data twice,
/// CFAVML guarantees that the output buffer will never be read from, so it is safe to provide
/// uninitialized buffers for the result, this is what the `WriteOnlyBuffer` trait is about.
///
/// ```rust
/// use core::mem::MaybeUninit;
///
/// let lhs = [1.0, -1.0, 0.5, 1.0];
/// let rhs = [1.0, 2.5, 0.5, -2.0];
///
/// let mut result = Vec::with_capacity(4);
/// unsafe { result.set_len(4) };
/// cfavml::lt_vertical(&lhs, &rhs, &mut result);
///
/// let result = unsafe { core::mem::transmute::<Vec<MaybeUninit<f32>>, Vec<f32>>(result) };
/// assert_eq!(result, [0.0, 1.0, 0.0, 0.0]);
/// ```
///
/// ### Projecting Vectors
///
/// CFAVML allows for working over a wide variety of buffers for applications, projection is effectively
/// broadcasting of two input buffers implementing `IntoMemLoader<T>`.
///
/// By default, you can provide _two slices_, _one slice and a broadcast value_, or _two broadcast values_,
/// which exhibit the standard behaviour as you might expect.
///
/// When providing two slices as inputs they cannot be projected to a buffer
/// that is larger their input sizes by default. This means providing two slices
/// of `128` elements in length must take a result buffer of `128` elements in length.
///
/// ### Implementation Pseudocode
///
/// _This is the logic of the routine being called._
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
/// - `NaN < 1.0 -> false`
/// - `NaN < NaN -> false`
///
/// # Panics
///
/// If vectors `a` and `b` cannot be projected to the target size of `result`.
/// Note that the projection rules are tied to the `MemLoader` implementation.
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
/// Checks each element pair of elements from vectors `a` and `b` comparing if
/// element `a` is **_less than or equal to_** element `b`, storing the output as `1` (true)
/// or `0` (false) in `result`.
///
/// ### Things To Know
///
/// ###### Supported patterns
///
/// Unlike horizontal operations, vertical ops can execute over a much wider variety of data
/// depending on the `MemLoader` which is a trait used to control how inputs are projected and
/// read from before executing. For the most part, you don't need to worry about this outside of
/// knowing you can by default, pass one combination of:
///     
/// - `lhs: vector` and `rhs: vector`  
/// - `lhs: vector` and `rhs: broadcast value`  
/// - `lhs: broadcast value` and `rhs: vector`  
/// - `lhs: broadcast value` and `rhs: broadcast value`  
///   * Not really that useful and is just an artefact of the memory management system.
///
/// ###### Broadcast values
///
/// When a broadcast value is provided, CFAVML will stretch that value out to match the size of
/// the _result_ buffer (not the other input buffer!) this does not cost additional allocations
/// outside the result buffer itself.
///
/// This means the following is possible:
/// - `[0, 0, 0] + 1 == [1, 1, 1]`  w/result_buffer_len=3
/// - `0 + 1 == [1, 1, 1]`  w/result_buffer_len=3
/// - `1 + 1 == [1]`  w/result_buffer_len=1
///
/// ###### Masks
///
/// CFAVML follows the same pattern as numpy, which is it representing boolean results as
/// either `1` or `0` in the respective type. This allows you to do various bit manipulation
/// and arithmetic techniques for processing values within the vector.
///
/// ### Examples
///
/// ##### Two vectors
///
/// ```rust
/// let lhs = [-1.0, 2.3, 2.0, 1.0];
/// let rhs = [2.0, 0.7, 2.0, -2.0];
///
/// let mut mask = [0.0f32; 4];
/// cfavml::lte_vertical(&lhs, &rhs, &mut mask);
/// assert_eq!(mask, [1.0, 0.0, 1.0, 0.0]);  // Index 0 & 2 are lte to rhs.
///
/// // Now we can use it to zero any values that are greater.
/// let mut match_or_zeroes = [0.0f32; 4];
/// cfavml::mul_vertical(&lhs, &mask, &mut match_or_zeroes);
///
/// // Our original match is extracted and the rest are `0.0`
/// // For convenience, I've used `0.0` as the non-match value,
/// // but if you switch `mask` and `lhs` around you can get
/// // a `NaN` mask which may be more useful depending on application.
/// assert_eq!(match_or_zeroes, [-1.0, 0.0, 2.0, 0.0]);    
/// ```
///
/// ##### One vector & broadcast value
///
/// ```rust
/// let lhs = [2.0, f32::NAN, -1.0, -0.5];
///
/// let mut result = [0.0f32; 4];
/// cfavml::lte_vertical(&lhs, -0.5, &mut result);
/// assert_eq!(result, [0.0, 0.0, 1.0, 1.0]);  // NaN is always false on lte checks.
/// ```
///
/// ##### Two broadcast values
///
/// ```rust
/// let mut result = [0.0f32; 4];
/// cfavml::lte_vertical(5.0, 5.0, &mut result);
/// assert_eq!(result, [1.0; 4]);
/// ```
///
/// ##### With `MaybeUninit`
///
/// Often if you are working with new-allocations, you do not want to initialize the data twice,
/// CFAVML guarantees that the output buffer will never be read from, so it is safe to provide
/// uninitialized buffers for the result, this is what the `WriteOnlyBuffer` trait is about.
///
/// ```rust
/// use core::mem::MaybeUninit;
///
/// let lhs = [1.0, -1.0, 0.5, 1.0];
/// let rhs = [1.0, 2.5, 0.5, -2.0];
///
/// let mut result = Vec::with_capacity(4);
/// unsafe { result.set_len(4) };
/// cfavml::lte_vertical(&lhs, &rhs, &mut result);
///
/// let result = unsafe { core::mem::transmute::<Vec<MaybeUninit<f32>>, Vec<f32>>(result) };
/// assert_eq!(result, [1.0, 1.0, 1.0, 0.0]);
/// ```
///
/// ### Projecting Vectors
///
/// CFAVML allows for working over a wide variety of buffers for applications, projection is effectively
/// broadcasting of two input buffers implementing `IntoMemLoader<T>`.
///
/// By default, you can provide _two slices_, _one slice and a broadcast value_, or _two broadcast values_,
/// which exhibit the standard behaviour as you might expect.
///
/// When providing two slices as inputs they cannot be projected to a buffer
/// that is larger their input sizes by default. This means providing two slices
/// of `128` elements in length must take a result buffer of `128` elements in length.
///
/// ### Implementation Pseudocode
///
/// _This is the logic of the routine being called._
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
/// - `1.0 <= 1.0 -> true`
/// - `0.0 <= NaN -> false`
/// - `NaN <= 1.0 -> false`
/// - `NaN <= NaN -> false`
///
/// # Panics
///
/// If vectors `a` and `b` cannot be projected to the target size of `result`.
/// Note that the projection rules are tied to the `MemLoader` implementation.
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
/// Checks each element pair of elements from vectors `a` and `b` comparing if
/// element `a` is **_greater than_** element `b`, storing the output as `1` (true) or `0` (false)
/// in `result`.
///
/// ### Things To Know
///
/// ###### Supported patterns
///
/// Unlike horizontal operations, vertical ops can execute over a much wider variety of data
/// depending on the `MemLoader` which is a trait used to control how inputs are projected and
/// read from before executing. For the most part, you don't need to worry about this outside of
/// knowing you can by default, pass one combination of:
///     
/// - `lhs: vector` and `rhs: vector`  
/// - `lhs: vector` and `rhs: broadcast value`  
/// - `lhs: broadcast value` and `rhs: vector`  
/// - `lhs: broadcast value` and `rhs: broadcast value`  
///   * Not really that useful and is just an artefact of the memory management system.
///
/// ###### Broadcast values
///
/// When a broadcast value is provided, CFAVML will stretch that value out to match the size of
/// the _result_ buffer (not the other input buffer!) this does not cost additional allocations
/// outside the result buffer itself.
///
/// This means the following is possible:
/// - `[0, 0, 0] + 1 == [1, 1, 1]`  w/result_buffer_len=3
/// - `0 + 1 == [1, 1, 1]`  w/result_buffer_len=3
/// - `1 + 1 == [1]`  w/result_buffer_len=1
///
/// ###### Masks
///
/// CFAVML follows the same pattern as numpy, which is it representing boolean results as
/// either `1` or `0` in the respective type. This allows you to do various bit manipulation
/// and arithmetic techniques for processing values within the vector.
///
/// ### Examples
///
/// ##### Two vectors
///
/// ```rust
/// let lhs = [-1.0, 2.3, 2.0, 1.0];
/// let rhs = [2.0, 0.7, 2.0, -2.0];
///
/// let mut mask = [0.0f32; 4];
/// cfavml::gt_vertical(&lhs, &rhs, &mut mask);
/// assert_eq!(mask, [0.0, 1.0, 0.0, 1.0]);  // Index 1 & 3 are greater than the rhs.
///
/// // Now we can use it to zero any values that are less than or equal.
/// let mut match_or_zeroes = [0.0f32; 4];
/// cfavml::mul_vertical(&lhs, &mask, &mut match_or_zeroes);
///
/// // Our original match is extracted and the rest are `0.0`
/// // For convenience, I've used `0.0` as the non-match value,
/// // but if you switch `mask` and `lhs` around you can get
/// // a `NaN` mask which may be more useful depending on application.
/// assert_eq!(match_or_zeroes, [0.0, 2.3, 0.0, 1.0]);    
/// ```
///
/// ##### One vector & broadcast value
///
/// ```rust
/// let lhs = [2.0, f32::NAN, -1.0, -0.5];
///
/// let mut result = [0.0f32; 4];
/// cfavml::gt_vertical(&lhs, -0.5, &mut result);
/// assert_eq!(result, [1.0, 0.0, 0.0, 0.0]);  // NaN is always false on gt checks.
/// ```
///
/// ##### Two broadcast values
///
/// ```rust
/// let mut result = [0.0f32; 4];
/// cfavml::gt_vertical(6.0, 5.0, &mut result);
/// assert_eq!(result, [1.0; 4]);
/// ```
///
/// ##### With `MaybeUninit`
///
/// Often if you are working with new-allocations, you do not want to initialize the data twice,
/// CFAVML guarantees that the output buffer will never be read from, so it is safe to provide
/// uninitialized buffers for the result, this is what the `WriteOnlyBuffer` trait is about.
///
/// ```rust
/// use core::mem::MaybeUninit;
///
/// let lhs = [1.0, -1.0, 0.5, 1.0];
/// let rhs = [1.0, 2.5, 0.5, -2.0];
///
/// let mut result = Vec::with_capacity(4);
/// unsafe { result.set_len(4) };
/// cfavml::gt_vertical(&lhs, &rhs, &mut result);
///
/// let result = unsafe { core::mem::transmute::<Vec<MaybeUninit<f32>>, Vec<f32>>(result) };
/// assert_eq!(result, [0.0, 0.0, 0.0, 1.0]);
/// ```
///
/// ### Projecting Vectors
///
/// CFAVML allows for working over a wide variety of buffers for applications, projection is effectively
/// broadcasting of two input buffers implementing `IntoMemLoader<T>`.
///
/// By default, you can provide _two slices_, _one slice and a broadcast value_, or _two broadcast values_,
/// which exhibit the standard behaviour as you might expect.
///
/// When providing two slices as inputs they cannot be projected to a buffer
/// that is larger their input sizes by default. This means providing two slices
/// of `128` elements in length must take a result buffer of `128` elements in length.
///
/// ### Implementation Pseudocode
///
/// _This is the logic of the routine being called._
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
/// - `NaN > 1.0 -> false`
/// - `NaN > NaN -> false`
///
/// # Panics
///
/// If vectors `a` and `b` cannot be projected to the target size of `result`.
/// Note that the projection rules are tied to the `MemLoader` implementation.
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
/// Checks each element pair of elements from vectors `a` and `b` comparing if
/// element `a` is **_greater than or equal to_** element `b`, storing the output as `1` (true)
/// or `0` (false) in `result`.
///
/// ### Things To Know
///
/// ###### Supported patterns
///
/// Unlike horizontal operations, vertical ops can execute over a much wider variety of data
/// depending on the `MemLoader` which is a trait used to control how inputs are projected and
/// read from before executing. For the most part, you don't need to worry about this outside of
/// knowing you can by default, pass one combination of:
///     
/// - `lhs: vector` and `rhs: vector`  
/// - `lhs: vector` and `rhs: broadcast value`  
/// - `lhs: broadcast value` and `rhs: vector`  
/// - `lhs: broadcast value` and `rhs: broadcast value`  
///   * Not really that useful and is just an artefact of the memory management system.
///
/// ###### Broadcast values
///
/// When a broadcast value is provided, CFAVML will stretch that value out to match the size of
/// the _result_ buffer (not the other input buffer!) this does not cost additional allocations
/// outside the result buffer itself.
///
/// This means the following is possible:
/// - `[0, 0, 0] + 1 == [1, 1, 1]`  w/result_buffer_len=3
/// - `0 + 1 == [1, 1, 1]`  w/result_buffer_len=3
/// - `1 + 1 == [1]`  w/result_buffer_len=1
///
/// ###### Masks
///
/// CFAVML follows the same pattern as numpy, which is it representing boolean results as
/// either `1` or `0` in the respective type. This allows you to do various bit manipulation
/// and arithmetic techniques for processing values within the vector.
///
/// ### Examples
///
/// ##### Two vectors
///
/// ```rust
/// let lhs = [-1.0, 2.3, 2.0, 1.0];
/// let rhs = [2.0, 0.7, 2.0, -2.0];
///
/// let mut mask = [0.0f32; 4];
/// cfavml::gte_vertical(&lhs, &rhs, &mut mask);
/// assert_eq!(mask, [0.0, 1.0, 1.0, 1.0]);  // Index 1, 2 & 3 are greater than or eq to rhs.
///
/// // Now we can use it to zero any values that are less than the rhs.
/// let mut match_or_zeroes = [0.0f32; 4];
/// cfavml::mul_vertical(&lhs, &mask, &mut match_or_zeroes);
///
/// // Our original match is extracted and the rest are `0.0`
/// // For convenience, I've used `0.0` as the non-match value,
/// // but if you switch `mask` and `lhs` around you can get
/// // a `NaN` mask which may be more useful depending on application.
/// assert_eq!(match_or_zeroes, [0.0, 2.3, 2.0, 1.0]);    
/// ```
///
/// ##### One vector & broadcast value
///
/// ```rust
/// let lhs = [2.0, f32::NAN, -1.0, -0.5];
///
/// let mut result = [0.0f32; 4];
/// cfavml::gte_vertical(&lhs, -0.5, &mut result);
/// assert_eq!(result, [1.0, 0.0, 0.0, 1.0]);  // NaN is always false on gte checks.
/// ```
///
/// ##### Two broadcast values
///
/// ```rust
/// let mut result = [0.0f32; 4];
/// cfavml::gte_vertical(5.0, 5.0, &mut result);
/// assert_eq!(result, [1.0; 4]);
/// ```
///
/// ##### With `MaybeUninit`
///
/// Often if you are working with new-allocations, you do not want to initialize the data twice,
/// CFAVML guarantees that the output buffer will never be read from, so it is safe to provide
/// uninitialized buffers for the result, this is what the `WriteOnlyBuffer` trait is about.
///
/// ```rust
/// use core::mem::MaybeUninit;
///
/// let lhs = [1.0, -1.0, 0.5, 1.0];
/// let rhs = [1.0, 2.5, 0.5, -2.0];
///
/// let mut result = Vec::with_capacity(4);
/// unsafe { result.set_len(4) };
/// cfavml::gte_vertical(&lhs, &rhs, &mut result);
///
/// let result = unsafe { core::mem::transmute::<Vec<MaybeUninit<f32>>, Vec<f32>>(result) };
/// assert_eq!(result, [1.0, 0.0, 1.0, 1.0]);
/// ```
///
/// ### Projecting Vectors
///
/// CFAVML allows for working over a wide variety of buffers for applications, projection is effectively
/// broadcasting of two input buffers implementing `IntoMemLoader<T>`.
///
/// By default, you can provide _two slices_, _one slice and a broadcast value_, or _two broadcast values_,
/// which exhibit the standard behaviour as you might expect.
///
/// When providing two slices as inputs they cannot be projected to a buffer
/// that is larger their input sizes by default. This means providing two slices
/// of `128` elements in length must take a result buffer of `128` elements in length.
///
/// ### Implementation Pseudocode
///
/// _This is the logic of the routine being called._
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
/// - `NaN >= 1.0 -> false`
/// - `NaN >= NaN -> false`
///
/// # Panics
///
/// If vectors `a` and `b` cannot be projected to the target size of `result`.
/// Note that the projection rules are tied to the `MemLoader` implementation.
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

/// Performs an element wise addition of two input buffers `a` and `b` that can
/// be projected to the desired output size of `result`.
///
/// ### Things To Know
///
/// ###### Supported patterns
///
/// Unlike horizontal operations, vertical ops can execute over a much wider variety of data
/// depending on the `MemLoader` which is a trait used to control how inputs are projected and
/// read from before executing. For the most part, you don't need to worry about this outside of
/// knowing you can by default, pass one combination of:
///     
/// - `lhs: vector` and `rhs: vector`  
/// - `lhs: vector` and `rhs: broadcast value`  
/// - `lhs: broadcast value` and `rhs: vector`  
/// - `lhs: broadcast value` and `rhs: broadcast value`  
///   * Not really that useful and is just an artefact of the memory management system.
///
/// ###### Broadcast values
///
/// When a broadcast value is provided, CFAVML will stretch that value out to match the size of
/// the _result_ buffer (not the other input buffer!) this does not cost additional allocations
/// outside the result buffer itself.
///
/// This means the following is possible:
/// - `[0, 0, 0] + 1 == [1, 1, 1]`  w/result_buffer_len=3
/// - `0 + 1 == [1, 1, 1]`  w/result_buffer_len=3
/// - `1 + 1 == [1]`  w/result_buffer_len=1
///
/// ### Examples
///
/// ##### Two vectors
///
/// ```rust
/// let lhs = [-1.0, 2.3, 2.0, 1.0];
/// let rhs = [2.0, 0.7, 2.0, -2.0];
///
/// let mut result = [0.0f32; 4];
/// cfavml::add_vertical(&lhs, &rhs, &mut result);
/// assert_eq!(result, [1.0, 3.0, 4.0, -1.0]);
/// ```
///
/// ##### One vector & broadcast value
///
/// ```rust
/// let lhs = [2.0, 0.5, -1.0, -0.5];
///
/// let mut result = [0.0f32; 4];
/// cfavml::add_vertical(&lhs, -0.5, &mut result);
/// assert_eq!(result, [1.5, 0.0, -1.5, -1.0]);
/// ```
///
/// ##### Two broadcast values
///
/// ```rust
/// let mut result = [0.0f32; 4];
/// cfavml::add_vertical(5.0, 5.0, &mut result);
/// assert_eq!(result, [10.0; 4]);
/// ```
///
/// ##### With `MaybeUninit`
///
/// Often if you are working with new-allocations, you do not want to initialize the data twice,
/// CFAVML guarantees that the output buffer will never be read from, so it is safe to provide
/// uninitialized buffers for the result, this is what the `WriteOnlyBuffer` trait is about.
///
/// ```rust
/// use core::mem::MaybeUninit;
///
/// let lhs = [1.0, -1.0, 0.5, 1.0];
/// let rhs = [1.0, 2.5, 0.5, -2.0];
///
/// let mut result = Vec::with_capacity(4);
/// unsafe { result.set_len(4) };
/// cfavml::add_vertical(&lhs, &rhs, &mut result);
///
/// let result = unsafe { core::mem::transmute::<Vec<MaybeUninit<f32>>, Vec<f32>>(result) };
/// assert_eq!(result, [2.0, 1.5, 1.0, -1.0]);
/// ```
///
/// ### Projecting Vectors
///
/// CFAVML allows for working over a wide variety of buffers for applications, projection is effectively
/// broadcasting of two input buffers implementing `IntoMemLoader<T>`.
///
/// By default, you can provide _two slices_, _one slice and a broadcast value_, or _two broadcast values_,
/// which exhibit the standard behaviour as you might expect.
///
/// When providing two slices as inputs they cannot be projected to a buffer
/// that is larger their input sizes by default. This means providing two slices
/// of `128` elements in length must take a result buffer of `128` elements in length.
///
/// ### Implementation Pseudocode
///
/// _This is the logic of the routine being called._
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
/// # Panics
///
/// If vectors `a` and `b` cannot be projected to the target size of `result`.
/// Note that the projection rules are tied to the `MemLoader` implementation.
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

/// Performs an element wise subtraction of two input buffers `a` and `b` that can
/// be projected to the desired output size of `result`.
///
/// ### Things To Know
///
/// ###### Supported patterns
///
/// Unlike horizontal operations, vertical ops can execute over a much wider variety of data
/// depending on the `MemLoader` which is a trait used to control how inputs are projected and
/// read from before executing. For the most part, you don't need to worry about this outside of
/// knowing you can by default, pass one combination of:
///     
/// - `lhs: vector` and `rhs: vector`  
/// - `lhs: vector` and `rhs: broadcast value`  
/// - `lhs: broadcast value` and `rhs: vector`  
/// - `lhs: broadcast value` and `rhs: broadcast value`  
///   * Not really that useful and is just an artefact of the memory management system.
///
/// ###### Broadcast values
///
/// When a broadcast value is provided, CFAVML will stretch that value out to match the size of
/// the _result_ buffer (not the other input buffer!) this does not cost additional allocations
/// outside the result buffer itself.
///
/// This means the following is possible:
/// - `[0, 0, 0] + 1 == [1, 1, 1]`  w/result_buffer_len=3
/// - `0 + 1 == [1, 1, 1]`  w/result_buffer_len=3
/// - `1 + 1 == [1]`  w/result_buffer_len=1
///
/// ### Examples
///
/// ##### Two vectors
///
/// ```rust
/// let lhs = [-1.0, 2.3, 2.0, 1.0];
/// let rhs = [2.0, 0.7, 2.0, -2.0];
///
/// let mut result = [0.0f32; 4];
/// cfavml::sub_vertical(&lhs, &rhs, &mut result);
/// assert_eq!(result, [-3.0, 1.5999999, 0.0, 3.0]);
/// ```
///
/// ##### One vector & broadcast value
///
/// ```rust
/// let lhs = [2.0, 0.5, -1.0, -0.5];
///
/// let mut result = [0.0f32; 4];
/// cfavml::sub_vertical(&lhs, -0.5, &mut result);
/// assert_eq!(result, [2.5, 1.0, -0.5, 0.0]);
/// ```
///
/// ##### Two broadcast values
///
/// ```rust
/// let mut result = [0.0f32; 4];
/// cfavml::sub_vertical(5.0, 5.0, &mut result);
/// assert_eq!(result, [0.0; 4]);
/// ```
///
/// ##### With `MaybeUninit`
///
/// Often if you are working with new-allocations, you do not want to initialize the data twice,
/// CFAVML guarantees that the output buffer will never be read from, so it is safe to provide
/// uninitialized buffers for the result, this is what the `WriteOnlyBuffer` trait is about.
///
/// ```rust
/// use core::mem::MaybeUninit;
///
/// let lhs = [1.0, -1.0, 0.5, 1.0];
/// let rhs = [1.0, 2.5, 0.5, -2.0];
///
/// let mut result = Vec::with_capacity(4);
/// unsafe { result.set_len(4) };
/// cfavml::sub_vertical(&lhs, &rhs, &mut result);
///
/// let result = unsafe { core::mem::transmute::<Vec<MaybeUninit<f32>>, Vec<f32>>(result) };
/// assert_eq!(result, [0.0, -3.5, 0.0, 3.0]);
/// ```
///
/// ### Projecting Vectors
///
/// CFAVML allows for working over a wide variety of buffers for applications, projection is effectively
/// broadcasting of two input buffers implementing `IntoMemLoader<T>`.
///
/// By default, you can provide _two slices_, _one slice and a broadcast value_, or _two broadcast values_,
/// which exhibit the standard behaviour as you might expect.
///
/// When providing two slices as inputs they cannot be projected to a buffer
/// that is larger their input sizes by default. This means providing two slices
/// of `128` elements in length must take a result buffer of `128` elements in length.
///
/// ### Implementation Pseudocode
///
/// _This is the logic of the routine being called._
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
/// # Panics
///
/// If vectors `a` and `b` cannot be projected to the target size of `result`.
/// Note that the projection rules are tied to the `MemLoader` implementation.
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

/// Performs an element wise multiply of two input buffers `a` and `b` that can
/// be projected to the desired output size of `result`.
///
/// ### Things To Know
///
/// ###### Supported patterns
///
/// Unlike horizontal operations, vertical ops can execute over a much wider variety of data
/// depending on the `MemLoader` which is a trait used to control how inputs are projected and
/// read from before executing. For the most part, you don't need to worry about this outside of
/// knowing you can by default, pass one combination of:
///     
/// - `lhs: vector` and `rhs: vector`  
/// - `lhs: vector` and `rhs: broadcast value`  
/// - `lhs: broadcast value` and `rhs: vector`  
/// - `lhs: broadcast value` and `rhs: broadcast value`  
///   * Not really that useful and is just an artefact of the memory management system.
///
/// ###### Broadcast values
///
/// When a broadcast value is provided, CFAVML will stretch that value out to match the size of
/// the _result_ buffer (not the other input buffer!) this does not cost additional allocations
/// outside the result buffer itself.
///
/// This means the following is possible:
/// - `[0, 0, 0] + 1 == [1, 1, 1]`  w/result_buffer_len=3
/// - `0 + 1 == [1, 1, 1]`  w/result_buffer_len=3
/// - `1 + 1 == [1]`  w/result_buffer_len=1
///
/// ### Examples
///
/// ##### Two vectors
///
/// ```rust
/// let lhs = [-1.0, 2.3, 2.0, 1.0];
/// let rhs = [2.0, 0.7, 2.0, -2.0];
///
/// let mut result = [0.0f32; 4];
/// cfavml::mul_vertical(&lhs, &rhs, &mut result);
/// assert_eq!(result, [-2.0, 1.6099999, 4.0, -2.0]);
/// ```
///
/// ##### One vector & broadcast value
///
/// ```rust
/// let lhs = [2.0, 0.5, -1.0, -0.5];
///
/// let mut result = [0.0f32; 4];
/// cfavml::mul_vertical(&lhs, -0.5, &mut result);
/// assert_eq!(result, [-1.0, -0.25, 0.5, 0.25]);
/// ```
///
/// ##### Two broadcast values
///
/// ```rust
/// let mut result = [0.0f32; 4];
/// cfavml::mul_vertical(5.0, 5.0, &mut result);
/// assert_eq!(result, [25.0; 4]);
/// ```
///
/// ##### With `MaybeUninit`
///
/// Often if you are working with new-allocations, you do not want to initialize the data twice,
/// CFAVML guarantees that the output buffer will never be read from, so it is safe to provide
/// uninitialized buffers for the result, this is what the `WriteOnlyBuffer` trait is about.
///
/// ```rust
/// use core::mem::MaybeUninit;
///
/// let lhs = [1.0, -1.0, 0.5, 1.0];
/// let rhs = [1.0, 2.5, 0.5, -2.0];
///
/// let mut result = Vec::with_capacity(4);
/// unsafe { result.set_len(4) };
/// cfavml::mul_vertical(&lhs, &rhs, &mut result);
///
/// let result = unsafe { core::mem::transmute::<Vec<MaybeUninit<f32>>, Vec<f32>>(result) };
/// assert_eq!(result, [1.0, -2.5, 0.25, -2.0]);
/// ```
///
/// ### Projecting Vectors
///
/// CFAVML allows for working over a wide variety of buffers for applications, projection is effectively
/// broadcasting of two input buffers implementing `IntoMemLoader<T>`.
///
/// By default, you can provide _two slices_, _one slice and a broadcast value_, or _two broadcast values_,
/// which exhibit the standard behaviour as you might expect.
///
/// When providing two slices as inputs they cannot be projected to a buffer
/// that is larger their input sizes by default. This means providing two slices
/// of `128` elements in length must take a result buffer of `128` elements in length.
///
/// ### Implementation Pseudocode
///
/// _This is the logic of the routine being called._
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
/// # Panics
///
/// If vectors `a` and `b` cannot be projected to the target size of `result`.
/// Note that the projection rules are tied to the `MemLoader` implementation.
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

/// Performs an element wise division of two input buffers `a` and `b` that can
/// be projected to the desired output size of `result`.
///
/// ## WARNING - You probably don't want to divide!
///
/// If you are attempting to divide a vector by a broadcast value or vice versa, you are
/// infinitely better off calculating the inverse and using a _multiply_ operation instead.
///
/// **This routine will not do this for you** so you are potentially missing out on a large
/// chunk of performance.
///
/// ```rust
/// let example = [1.0, 2.0, 3.0];
/// let mut result = [0.0; 3];
/// // BAD! Don't do this!
/// cfavml::div_vertical(&example, 2.0, &mut result);
/// assert_eq!(result, [0.5, 1.0, 1.5]);
///
/// // Do this instead! It is infinitely faster! (Well not quite but trust me)
/// cfavml::mul_vertical(&example, 1.0 / 2.0, &mut result);
/// assert_eq!(result, [0.5, 1.0, 1.5]);
/// ```
///
/// ### Things To Know
///
/// ###### Supported patterns
///
/// Unlike horizontal operations, vertical ops can execute over a much wider variety of data
/// depending on the `MemLoader` which is a trait used to control how inputs are projected and
/// read from before executing. For the most part, you don't need to worry about this outside of
/// knowing you can by default, pass one combination of:
///     
/// - `lhs: vector` and `rhs: vector`  
/// - `lhs: vector` and `rhs: broadcast value`  
/// - `lhs: broadcast value` and `rhs: vector`  
/// - `lhs: broadcast value` and `rhs: broadcast value`  
///   * Not really that useful and is just an artefact of the memory management system.
///
/// ###### Broadcast values
///
/// When a broadcast value is provided, CFAVML will stretch that value out to match the size of
/// the _result_ buffer (not the other input buffer!) this does not cost additional allocations
/// outside the result buffer itself.
///
/// This means the following is possible:
/// - `[0, 0, 0] + 1 == [1, 1, 1]`  w/result_buffer_len=3
/// - `0 + 1 == [1, 1, 1]`  w/result_buffer_len=3
/// - `1 + 1 == [1]`  w/result_buffer_len=1
///
/// ### Examples
///
/// ##### Two vectors
///
/// ```rust
/// let lhs = [1.0, 2.0, 4.0, 1.0];
/// let rhs = [2.0, 0.8, 2.0, -2.0];
///
/// let mut result = [0.0f32; 4];
/// cfavml::div_vertical(&lhs, &rhs, &mut result);
/// assert_eq!(result, [0.5, 2.5, 2.0, -0.5]);
/// ```
///
/// ##### One vector & broadcast value
///
/// ```rust
/// let lhs = [2.0, 0.5, -1.0, -0.5];
///
/// let mut result = [0.0f32; 4];
/// cfavml::div_vertical(&lhs, -0.5, &mut result);
/// assert_eq!(result, [-4.0, -1.0, 2.0, 1.0]);
/// ```
///
/// ##### Two broadcast values
///
/// ```rust
/// let mut result = [0.0f32; 4];
/// cfavml::div_vertical(5.0, 5.0, &mut result);
/// assert_eq!(result, [1.0; 4]);
/// ```
///
/// ##### With `MaybeUninit`
///
/// Often if you are working with new-allocations, you do not want to initialize the data twice,
/// CFAVML guarantees that the output buffer will never be read from, so it is safe to provide
/// uninitialized buffers for the result, this is what the `WriteOnlyBuffer` trait is about.
///
/// ```rust
/// use core::mem::MaybeUninit;
///
/// let lhs = [1.0, -1.0, 0.5, 1.0];
/// let rhs = [1.0, 2.5, 0.5, -2.0];
///
/// let mut result = Vec::with_capacity(4);
/// unsafe { result.set_len(4) };
/// cfavml::div_vertical(&lhs, &rhs, &mut result);
///
/// let result = unsafe { core::mem::transmute::<Vec<MaybeUninit<f32>>, Vec<f32>>(result) };
/// assert_eq!(result, [1.0, -0.4, 1.0, -0.5]);
/// ```
///
/// ### Projecting Vectors
///
/// CFAVML allows for working over a wide variety of buffers for applications, projection is effectively
/// broadcasting of two input buffers implementing `IntoMemLoader<T>`.
///
/// By default, you can provide _two slices_, _one slice and a broadcast value_, or _two broadcast values_,
/// which exhibit the standard behaviour as you might expect.
///
/// When providing two slices as inputs they cannot be projected to a buffer
/// that is larger their input sizes by default. This means providing two slices
/// of `128` elements in length must take a result buffer of `128` elements in length.
///
/// ### Implementation Pseudocode
///
/// _This is the logic of the routine being called._
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
/// # Panics
///
/// If vectors `a` and `b` cannot be projected to the target size of `result`.
/// Note that the projection rules are tied to the `MemLoader` implementation.
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
