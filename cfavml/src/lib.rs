#![allow(internal_features)]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]
#![cfg_attr(feature = "nightly", feature(core_intrinsics))]
#![cfg_attr(
    all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"),
    feature(avx512_target_feature)
)]
#![cfg_attr(
    all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"),
    feature(stdarch_x86_avx512)
)]
#![doc = include_str!("../README.md")]

pub mod danger;
pub mod dispatch;
pub mod math;

pub mod buffer;
pub mod mem_loader;
mod safe_function_ops;
pub mod safe_trait_agg_ops;
pub mod safe_trait_arithmetic_ops;
pub mod safe_trait_cmp_ops;
pub mod safe_trait_distance_ops;
pub mod safe_trait_misc_float_ops;
#[cfg(test)]
mod test_utils;

pub use self::safe_function_ops::*;
