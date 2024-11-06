pub mod complex_ops;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod impl_avx2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod impl_avx2fma;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub mod impl_avx512;

pub mod impl_fallback;
#[cfg(target_arch = "aarch64")]
pub mod impl_neon;
#[cfg(test)]
pub mod impl_test;
#[cfg(test)]
mod test_suite;
