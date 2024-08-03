mod core_simd_api;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod impl_avx2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod impl_avx2_fma;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
mod impl_avx512;
mod impl_fallback;
#[cfg(target_arch = "aarch64")]
mod impl_neon;
mod op_cosine;
mod op_dot_product;
mod op_euclidean;
mod op_max;
mod op_min;
mod op_norm;
mod op_sum;
mod op_vector_x_value;
mod op_vector_x_vector;

mod export_arithmetic_ops;
mod export_distance_ops;
mod export_min_max_sum_norm;
#[cfg(test)]
mod impl_test;
#[cfg(test)]
mod test_suite;

pub use self::core_simd_api::{DenseLane, SimdRegister};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::export_arithmetic_ops::arithmetic_ops_avx2::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::export_arithmetic_ops::arithmetic_ops_avx512::*;
pub use self::export_arithmetic_ops::arithmetic_ops_fallback::*;
#[cfg(target_arch = "aarch64")]
pub use self::export_arithmetic_ops::arithmetic_ops_neon::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::export_distance_ops::distance_ops_avx2::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::export_distance_ops::distance_ops_avx2_fma::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::export_distance_ops::distance_ops_avx512::*;
pub use self::export_distance_ops::distance_ops_fallback::*;
#[cfg(target_arch = "aarch64")]
pub use self::export_distance_ops::distance_ops_neon::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::export_min_max_sum_norm::min_max_sum_ops_avx2::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::export_min_max_sum_norm::min_max_sum_ops_avx512::*;
pub use self::export_min_max_sum_norm::min_max_sum_ops_fallback::*;
#[cfg(target_arch = "aarch64")]
pub use self::export_min_max_sum_norm::min_max_sum_ops_neon::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::impl_avx2::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::impl_avx2_fma::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::impl_avx512::*;
pub use self::impl_fallback::*;
#[cfg(target_arch = "aarch64")]
pub use self::impl_neon::*;
#[cfg(test)]
pub(crate) use self::op_cosine::cosine;
pub use self::op_cosine::generic_cosine;
pub use self::op_dot_product::generic_dot_product;
pub use self::op_euclidean::generic_euclidean;
pub use self::op_max::{generic_max_horizontal, generic_max_vertical};
pub use self::op_min::{generic_min_horizontal, generic_min_vertical};
pub use self::op_norm::generic_squared_norm;
pub use self::op_sum::generic_sum_horizontal;
pub use self::op_vector_x_value::{
    generic_add_value,
    generic_div_value,
    generic_mul_value,
    generic_sub_value,
};
pub use self::op_vector_x_vector::{
    generic_add_vector,
    generic_div_vector,
    generic_mul_vector,
    generic_sub_vector,
};

#[allow(non_snake_case)]
pub(crate) const fn _MM_SHUFFLE(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}
