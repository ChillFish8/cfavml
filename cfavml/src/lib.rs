#![allow(internal_features)]
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

pub mod danger;
pub mod math;

#[cfg(test)]
mod test_utils;
