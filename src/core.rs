/// A set of SIMD accelerated ndarray operations on 1-dimensional arrays.
pub trait AcceleratedLinearOps<T> {
    /// Performs a horizontal sum across the array.
    fn sum(&self) -> T;

    /// Computes the horizontal mean across the array.
    fn mean(&self) -> T;

    /// Finds the minimum value in the array.
    fn min(&self) -> T;

    /// Finds the maximum value in the array.
    fn max(&self) -> T;

    /// Computes the L2 norm of the vector.
    fn l2(&self) -> T;

    /// Computes the squared L2 norm of the vector.
    fn l2_squared(&self) -> T;

    /// Computes the dot product of the given vector.
    fn dot(&self, other: &Self) -> T;

    /// Computes the cosine distance of `self` and the `other` vector.
    fn cosine(&self, other: &Self) -> T;

    /// Computes the Euclidean distance of `self` and the `other` vector.
    fn euclidean(&self, other: &Self) -> T;

    /// Computes the squared Euclidean distance of `self` and the `other` vector.
    fn euclidean_squared(&self, other: &Self) -> T;
}

/// Accelerated arithmetic operations over another vector or single value.
///
/// Due to orphan rules and conflicts, we cannot override the inbuilt arithmetic
/// traits, so this is our best attempt at providing a somewhat ergonomic API.
pub trait AcceleratedArithmetic<T = Self> {
    /// Adds the `right` element to `self`.
    fn add_fast(&mut self, right: &T);
    /// Subtracts the `right` element from `self`.
    fn sub_fast(&mut self, right: &T);
    /// Multiplies `self` by the `right` element.
    fn mul_fast(&mut self, right: &T);
    /// Divides `self` by the `right` element.
    fn div_fast(&mut self, right: &T);
}

mod enabled_arch {
    pub(crate) const USE_DEFAULT: usize = 0;
    pub(crate) const AVX2_ENABLED: usize = 1;
    pub(crate) const AVX2_FMA_ENABLED: usize = 2;
    pub(crate) const AVX512_ENABLED: usize = 3;

    fn get_enabled() {}
}
