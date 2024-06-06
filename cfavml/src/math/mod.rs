mod default;
#[cfg(feature = "nightly")]
mod fast_math;

pub use default::StdMath;
#[cfg(feature = "nightly")]
pub use fast_math::FastMath;

#[cfg(not(feature = "nightly"))]
pub type AutoMath = StdMath;
#[cfg(feature = "nightly")]
pub type AutoMath = FastMath;

/// Core simple math operations that can be adjusted for certain features
/// or architectures.
pub trait Math<T> {
    /// Returns the equivalent zero value.
    fn zero() -> T;

    /// Returns the equivalent 1.0 value.
    fn one() -> T;

    /// The maximum value that the value can hold.
    fn max() -> T;

    /// The minimum value that the value can hold.
    fn min() -> T;

    /// Returns the equivalent 1.0 value.
    fn sqrt(a: T) -> T;

    /// Returns the abs of the value.
    fn abs(a: T) -> T;

    /// Returns if the two values are equal.
    fn cmp_eq(a: T, b: T) -> bool;

    /// Returns the minimum of the two values
    fn cmp_min(a: T, b: T) -> T;

    /// Returns the maximum of the two values
    fn cmp_max(a: T, b: T) -> T;

    /// `a + b`
    fn add(a: T, b: T) -> T;

    /// `a - b`
    fn sub(a: T, b: T) -> T;

    /// `a * b`
    fn mul(a: T, b: T) -> T;

    /// `a / b`
    fn div(a: T, b: T) -> T;
}
