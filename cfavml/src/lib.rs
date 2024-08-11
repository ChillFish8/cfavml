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
mod safe_arithmetic_ops;
mod safe_distance_ops;
mod safe_min_max_sum_ops;
mod safe_norm_ops;
#[cfg(test)]
mod test_utils;

pub use self::safe_arithmetic_ops::*;
pub use self::safe_distance_ops::*;
pub use self::safe_min_max_sum_ops::*;
pub use self::safe_norm_ops::*;
