#![allow(clippy::missing_transmute_annotations)]

mod core_simd_api;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod impl_avx2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod impl_avx2fma;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
mod impl_avx512;
mod impl_fallback;
#[cfg(target_arch = "aarch64")]
mod impl_neon;
mod op_arithmetic_value;
mod op_arithmetic_vector;
mod op_cmp_max;
mod op_cmp_min;
mod op_cosine;
mod op_dot;
mod op_euclidean;
mod op_norm;
mod op_sum;

mod core_routine_boilerplate;
pub mod export_agg_ops;
pub mod export_arithmetic_ops;
pub mod export_cmp_ops;
pub mod export_distance_ops;
#[cfg(test)]
mod impl_test;
mod op_cmp_value;
mod op_cmp_vector;
#[cfg(test)]
mod test_suite;

pub use self::core_simd_api::{DenseLane, SimdRegister};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::impl_avx2::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::impl_avx2fma::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::impl_avx512::*;
pub use self::impl_fallback::*;
#[cfg(target_arch = "aarch64")]
pub use self::impl_neon::*;
pub use self::op_arithmetic_value::{
    generic_add_value,
    generic_div_value,
    generic_mul_value,
    generic_sub_value,
};
pub use self::op_arithmetic_vector::{
    generic_add_vector,
    generic_div_vector,
    generic_mul_vector,
    generic_sub_vector,
};
pub use self::op_cmp_max::{
    generic_cmp_max,
    generic_cmp_max_vertical,
};
pub use self::op_cmp_min::{
    generic_cmp_min,
    generic_cmp_min_value,
    generic_cmp_min_vector,
};
pub use self::op_cmp_value::{
    generic_cmp_eq_value,
    generic_cmp_gt_value,
    generic_cmp_gte_value,
    generic_cmp_lt_value,
    generic_cmp_lte_value,
    generic_cmp_neq_value,
};
pub use self::op_cmp_vector::{
    generic_cmp_eq_vector,
    generic_cmp_gt_vector,
    generic_cmp_gte_vector,
    generic_cmp_lt_vector,
    generic_cmp_lte_vector,
    generic_cmp_neq_vector,
};
#[cfg(test)]
pub(crate) use self::op_cosine::cosine;
pub use self::op_cosine::generic_cosine;
pub use self::op_dot::generic_dot;
pub use self::op_euclidean::generic_squared_euclidean;
pub use self::op_norm::generic_squared_norm;
pub use self::op_sum::generic_sum;

#[allow(non_snake_case)]
pub(crate) const fn _MM_SHUFFLE(z: u32, y: u32, x: u32, w: u32) -> i32 {
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}
