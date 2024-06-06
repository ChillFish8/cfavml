#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod f32_avx2_angular_hyperplane;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod f32_avx2_cosine;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod f32_avx2_dot_product;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod f32_avx2_euclidean;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod f32_avx2_euclidean_hyperplane;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod f32_avx2_max;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod f32_avx2_min;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod f32_avx2_norm;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod f32_avx2_sum;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod f32_avx2_vector_x_value;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod f32_avx2_vector_x_vector;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
mod f32_avx512_angular_hyperplane;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
mod f32_avx512_cosine;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
mod f32_avx512_dot_product;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
mod f32_avx512_euclidean;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
mod f32_avx512_euclidean_hyperplane;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
mod f32_avx512_max;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
mod f32_avx512_min;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
mod f32_avx512_norm;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
mod f32_avx512_sum;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
mod f32_avx512_vector_x_value;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
mod f32_avx512_vector_x_vector;
mod f32_fallback_angular_hyperplane;
mod f32_fallback_euclidean_hyperplane;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod f64_avx2_cosine;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod f64_avx2_dot_product;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod f64_avx2_euclidean;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod f64_avx2_max;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod f64_avx2_min;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod f64_avx2_norm;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod f64_avx2_sum;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod f64_avx2_vector_x_value;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod f64_avx2_vector_x_vector;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
mod f64_avx512_cosine;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
mod f64_avx512_dot_product;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
mod f64_avx512_euclidean;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
mod f64_avx512_max;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
mod f64_avx512_min;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
mod f64_avx512_norm;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
mod f64_avx512_sum;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
mod f64_avx512_vector_x_value;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
mod f64_avx512_vector_x_vector;
mod generic_fallback_cosine;
mod generic_fallback_dot_product;
mod generic_fallback_euclidean;
mod generic_fallback_max;
mod generic_fallback_min;
mod generic_fallback_sum;
mod generic_fallback_vector_x_value;
mod generic_fallback_vector_x_vector;
mod utils;

pub(crate) use utils::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::f32_avx2_angular_hyperplane::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::f32_avx2_cosine::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::f32_avx2_dot_product::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::f32_avx2_euclidean::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::f32_avx2_euclidean_hyperplane::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::f32_avx2_max::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::f32_avx2_min::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::f32_avx2_norm::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::f32_avx2_sum::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::f32_avx2_vector_x_value::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::f32_avx2_vector_x_vector::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::f32_avx512_angular_hyperplane::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::f32_avx512_cosine::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::f32_avx512_dot_product::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::f32_avx512_euclidean::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::f32_avx512_euclidean_hyperplane::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::f32_avx512_max::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::f32_avx512_min::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::f32_avx512_norm::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::f32_avx512_sum::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::f32_avx512_vector_x_value::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::f32_avx512_vector_x_vector::*;
pub use self::f32_fallback_angular_hyperplane::*;
pub use self::f32_fallback_euclidean_hyperplane::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::f64_avx2_cosine::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::f64_avx2_dot_product::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::f64_avx2_euclidean::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::f64_avx2_max::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::f64_avx2_min::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::f64_avx2_norm::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::f64_avx2_sum::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::f64_avx2_vector_x_value::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::f64_avx2_vector_x_vector::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::f64_avx512_cosine::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::f64_avx512_dot_product::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::f64_avx512_euclidean::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::f64_avx512_max::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::f64_avx512_min::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::f64_avx512_norm::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::f64_avx512_sum::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::f64_avx512_vector_x_value::*;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
pub use self::f64_avx512_vector_x_vector::*;
pub use self::generic_fallback_cosine::*;
pub use self::generic_fallback_dot_product::*;
pub use self::generic_fallback_euclidean::*;
pub use self::generic_fallback_max::*;
pub use self::generic_fallback_min::*;
pub use self::generic_fallback_sum::*;
pub use self::generic_fallback_vector_x_value::*;
pub use self::generic_fallback_vector_x_vector::*;
