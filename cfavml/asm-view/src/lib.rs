#![allow(clippy::missing_safety_doc)]

use cfavml::danger::*;

macro_rules! export_dense_op {
    ($t:ident, $im:ident, $op:ident, features = $($feat:expr $(,)?)*) => {
        paste::paste!{
            #[inline(never)]
            #[target_feature($(enable = $feat ,)*)]
            pub unsafe fn [<impl_ $im:lower _ $t _ $op>](a: &[$t], b: &[$t]) {
                let l1 = <$im as cfavml::danger::SimdRegister<$t>>::load_dense(a.as_ptr());
                let l2 = <$im as cfavml::danger::SimdRegister<$t>>::load_dense(b.as_ptr());
                let res = <$im as cfavml::danger::SimdRegister<$t>>::$op(l1, l2);
                std::hint::black_box(res);
            }
        }
    };
    ($t:ident, $im:ident, $op:ident) => {
        paste::paste!{
            #[inline(never)]
            pub unsafe fn [<impl_ $im:lower _ $t _ $op>](a: &[$t], b: &[$t]) {
                let l1 = <$im as cfavml::danger::SimdRegister<$t>>::load_dense(a.as_ptr());
                let l2 = <$im as cfavml::danger::SimdRegister<$t>>::load_dense(b.as_ptr());
                let res = <$im as cfavml::danger::SimdRegister<$t>>::$op(l1, l2);
                std::hint::black_box(res);
            }
        }
    };
}

macro_rules! export_distance_op {
    ($t:ident, $im:ident, $op:ident, features = $($feat:expr $(,)?)*) => {
        paste::paste!{
            #[inline(never)]
            #[target_feature($(enable = $feat ,)*)]
            pub unsafe fn [<impl_ $im:lower _ $t _ $op>](a: &[$t], b: &[$t]) {
                let res = $op::<_, $im, cfavml::math::AutoMath>(a.len(), a, b)  ;
                std::hint::black_box(res);
            }
        }
    };
    ($t:ident, $im:ident, $op:ident) => {
        paste::paste!{
            #[inline(never)]
            pub unsafe fn [<impl_ $im:lower _ $t _ $op>](a: &[$t], b: &[$t]) {
                let res = $op::<_, $im, cfavml::math::AutoMath>(a.len(), a, b)  ;
                std::hint::black_box(res);
            }
        }
    };
}

macro_rules! export_vector_x_vector_op {
    ($t:ident, $im:ident, $op:ident, features = $($feat:expr $(,)?)*) => {
        paste::paste!{
            #[inline(never)]
            #[target_feature($(enable = $feat ,)*)]
            pub unsafe fn [<impl_ $im:lower _ $t _ $op>](a: &[$t], b: &[$t], res: &mut [$t]) {
                let res = $op::<_, $im, cfavml::math::AutoMath, _>(a.len(), a, b, res)  ;
                std::hint::black_box(res);
            }
        }
    };
    ($t:ident, $im:ident, $op:ident) => {
        paste::paste!{
            #[inline(never)]
            pub unsafe fn [<impl_ $im:lower _ $t _ $op>](a: &[$t], b: &[$t], res: &mut [$t]) {
                let res = $op::<_, $im, cfavml::math::AutoMath, _>(a.len(), a, b, res)  ;
                std::hint::black_box(res);
            }
        }
    };
}

macro_rules! export_vector_x_value_op {
    ($t:ident, $im:ident, $op:ident, features = $($feat:expr $(,)?)*) => {
        paste::paste!{
            #[inline(never)]
            #[target_feature($(enable = $feat ,)*)]
            pub unsafe fn [<impl_ $im:lower _ $t _ $op>](value: $t, a: &[$t], res: &mut [$t]) {
                let res = $op::<_, $im, cfavml::math::AutoMath, _>(a.len(), value, a, res)  ;
                std::hint::black_box(res);
            }
        }
    };
    ($t:ident, $im:ident, $op:ident) => {
        paste::paste!{
            #[inline(never)]
            pub unsafe fn [<impl_ $im:lower _ $t _ $op>](value: $t, a: &[$t], res: &mut [$t]) {
                let res = $op::<_, $im, cfavml::math::AutoMath, _>(a.len(), value, a, res)  ;
                std::hint::black_box(res);
            }
        }
    };
}

export_dense_op!(i8, Fallback, mul_dense);
export_dense_op!(i8, Fallback, div_dense);
export_dense_op!(i8, Fallback, add_dense);
export_dense_op!(i8, Fallback, sub_dense);
export_dense_op!(i8, Fallback, max_dense);
export_dense_op!(i8, Fallback, min_dense);

export_dense_op!(i16, Fallback, mul_dense);
export_dense_op!(i16, Fallback, div_dense);
export_dense_op!(i16, Fallback, add_dense);
export_dense_op!(i16, Fallback, sub_dense);
export_dense_op!(i16, Fallback, max_dense);
export_dense_op!(i16, Fallback, min_dense);

export_dense_op!(i32, Fallback, mul_dense);
export_dense_op!(i32, Fallback, div_dense);
export_dense_op!(i32, Fallback, add_dense);
export_dense_op!(i32, Fallback, sub_dense);
export_dense_op!(i32, Fallback, max_dense);
export_dense_op!(i32, Fallback, min_dense);

export_dense_op!(i64, Fallback, mul_dense);
export_dense_op!(i64, Fallback, div_dense);
export_dense_op!(i64, Fallback, add_dense);
export_dense_op!(i64, Fallback, sub_dense);
export_dense_op!(i64, Fallback, max_dense);
export_dense_op!(i64, Fallback, min_dense);

export_dense_op!(f32, Fallback, mul_dense);
export_dense_op!(f32, Fallback, div_dense);
export_dense_op!(f32, Fallback, add_dense);
export_dense_op!(f32, Fallback, sub_dense);
export_dense_op!(f32, Fallback, max_dense);
export_dense_op!(f32, Fallback, min_dense);

export_dense_op!(f64, Fallback, mul_dense);
export_dense_op!(f64, Fallback, div_dense);
export_dense_op!(f64, Fallback, add_dense);
export_dense_op!(f64, Fallback, sub_dense);
export_dense_op!(f64, Fallback, max_dense);
export_dense_op!(f64, Fallback, min_dense);

export_distance_op!(i8, Fallback, generic_cosine);
export_distance_op!(i8, Fallback, generic_dot);
export_distance_op!(i8, Fallback, generic_squared_euclidean);
export_vector_x_vector_op!(i8, Fallback, generic_add_vector);
export_vector_x_vector_op!(i8, Fallback, generic_sub_vector);
export_vector_x_vector_op!(i8, Fallback, generic_mul_vector);
export_vector_x_vector_op!(i8, Fallback, generic_div_vector);
export_vector_x_value_op!(i8, Fallback, generic_add_value);
export_vector_x_value_op!(i8, Fallback, generic_sub_value);
export_vector_x_value_op!(i8, Fallback, generic_mul_value);
export_vector_x_value_op!(i8, Fallback, generic_div_value);

export_distance_op!(f32, Fallback, generic_cosine);
export_distance_op!(f32, Fallback, generic_dot);
export_distance_op!(f32, Fallback, generic_squared_euclidean);
export_vector_x_vector_op!(f32, Fallback, generic_add_vector);
export_vector_x_vector_op!(f32, Fallback, generic_sub_vector);
export_vector_x_vector_op!(f32, Fallback, generic_mul_vector);
export_vector_x_vector_op!(f32, Fallback, generic_div_vector);
export_vector_x_value_op!(f32, Fallback, generic_add_value);
export_vector_x_value_op!(f32, Fallback, generic_sub_value);
export_vector_x_value_op!(f32, Fallback, generic_mul_value);
export_vector_x_value_op!(f32, Fallback, generic_div_value);

export_distance_op!(f64, Fallback, generic_cosine);
export_distance_op!(f64, Fallback, generic_dot);
export_distance_op!(f64, Fallback, generic_squared_euclidean);
export_vector_x_vector_op!(f64, Fallback, generic_add_vector);
export_vector_x_vector_op!(f64, Fallback, generic_sub_vector);
export_vector_x_vector_op!(f64, Fallback, generic_mul_vector);
export_vector_x_vector_op!(f64, Fallback, generic_div_vector);
export_vector_x_value_op!(f64, Fallback, generic_add_value);
export_vector_x_value_op!(f64, Fallback, generic_sub_value);
export_vector_x_value_op!(f64, Fallback, generic_mul_value);
export_vector_x_value_op!(f64, Fallback, generic_div_value);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx2_ops {
    use super::*;

    export_dense_op!(i8, Avx2, mul_dense, features = "avx2");
    export_dense_op!(i8, Avx2, div_dense, features = "avx2");
    export_dense_op!(i8, Avx2, add_dense, features = "avx2");
    export_dense_op!(i8, Avx2, sub_dense, features = "avx2");
    export_dense_op!(i8, Avx2, max_dense, features = "avx2");
    export_dense_op!(i8, Avx2, min_dense, features = "avx2");

    export_dense_op!(i16, Avx2, mul_dense, features = "avx2");
    export_dense_op!(i16, Avx2, div_dense, features = "avx2");
    export_dense_op!(i16, Avx2, add_dense, features = "avx2");
    export_dense_op!(i16, Avx2, sub_dense, features = "avx2");
    export_dense_op!(i16, Avx2, max_dense, features = "avx2");
    export_dense_op!(i16, Avx2, min_dense, features = "avx2");

    export_dense_op!(i32, Avx2, mul_dense, features = "avx2");
    export_dense_op!(i32, Avx2, div_dense, features = "avx2");
    export_dense_op!(i32, Avx2, add_dense, features = "avx2");
    export_dense_op!(i32, Avx2, sub_dense, features = "avx2");
    export_dense_op!(i32, Avx2, max_dense, features = "avx2");
    export_dense_op!(i32, Avx2, min_dense, features = "avx2");

    export_dense_op!(i64, Avx2, mul_dense, features = "avx2");
    export_dense_op!(i64, Avx2, div_dense, features = "avx2");
    export_dense_op!(i64, Avx2, add_dense, features = "avx2");
    export_dense_op!(i64, Avx2, sub_dense, features = "avx2");
    export_dense_op!(i64, Avx2, max_dense, features = "avx2");
    export_dense_op!(i64, Avx2, min_dense, features = "avx2");

    export_dense_op!(f32, Avx2, mul_dense, features = "avx2");
    export_dense_op!(f32, Avx2, div_dense, features = "avx2");
    export_dense_op!(f32, Avx2, add_dense, features = "avx2");
    export_dense_op!(f32, Avx2, sub_dense, features = "avx2");
    export_dense_op!(f32, Avx2, max_dense, features = "avx2");
    export_dense_op!(f32, Avx2, min_dense, features = "avx2");

    export_dense_op!(f64, Avx2, mul_dense, features = "avx2");
    export_dense_op!(f64, Avx2, div_dense, features = "avx2");
    export_dense_op!(f64, Avx2, add_dense, features = "avx2");
    export_dense_op!(f64, Avx2, sub_dense, features = "avx2");
    export_dense_op!(f64, Avx2, max_dense, features = "avx2");
    export_dense_op!(f64, Avx2, min_dense, features = "avx2");

    export_distance_op!(i8, Avx2, generic_cosine, features = "avx2");
    export_distance_op!(i8, Avx2, generic_dot, features = "avx2");
    export_distance_op!(i8, Avx2, generic_squared_euclidean, features = "avx2");
    export_vector_x_vector_op!(i8, Avx2, generic_add_vector, features = "avx2");
    export_vector_x_vector_op!(i8, Avx2, generic_sub_vector, features = "avx2");
    export_vector_x_vector_op!(i8, Avx2, generic_mul_vector, features = "avx2");
    export_vector_x_vector_op!(i8, Avx2, generic_div_vector, features = "avx2");
    export_vector_x_value_op!(i8, Avx2, generic_add_value, features = "avx2");
    export_vector_x_value_op!(i8, Avx2, generic_sub_value, features = "avx2");
    export_vector_x_value_op!(i8, Avx2, generic_mul_value, features = "avx2");
    export_vector_x_value_op!(i8, Avx2, generic_div_value, features = "avx2");

    export_distance_op!(f32, Avx2, generic_cosine, features = "avx2");
    export_distance_op!(f32, Avx2, generic_dot, features = "avx2");
    export_distance_op!(f32, Avx2, generic_squared_euclidean, features = "avx2");
    export_vector_x_vector_op!(f32, Avx2, generic_add_vector, features = "avx2");
    export_vector_x_vector_op!(f32, Avx2, generic_sub_vector, features = "avx2");
    export_vector_x_vector_op!(f32, Avx2, generic_mul_vector, features = "avx2");
    export_vector_x_vector_op!(f32, Avx2, generic_div_vector, features = "avx2");
    export_vector_x_value_op!(f32, Avx2, generic_add_value, features = "avx2");
    export_vector_x_value_op!(f32, Avx2, generic_sub_value, features = "avx2");
    export_vector_x_value_op!(f32, Avx2, generic_mul_value, features = "avx2");
    export_vector_x_value_op!(f32, Avx2, generic_div_value, features = "avx2");

    export_distance_op!(f64, Avx2, generic_cosine, features = "avx2");
    export_distance_op!(f64, Avx2, generic_dot, features = "avx2");
    export_distance_op!(f64, Avx2, generic_squared_euclidean, features = "avx2");
    export_vector_x_vector_op!(f64, Avx2, generic_add_vector, features = "avx2");
    export_vector_x_vector_op!(f64, Avx2, generic_sub_vector, features = "avx2");
    export_vector_x_vector_op!(f64, Avx2, generic_mul_vector, features = "avx2");
    export_vector_x_vector_op!(f64, Avx2, generic_div_vector, features = "avx2");
    export_vector_x_value_op!(f64, Avx2, generic_add_value, features = "avx2");
    export_vector_x_value_op!(f64, Avx2, generic_sub_value, features = "avx2");
    export_vector_x_value_op!(f64, Avx2, generic_mul_value, features = "avx2");
    export_vector_x_value_op!(f64, Avx2, generic_div_value, features = "avx2");
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx2fma_ops {
    use super::*;

    export_distance_op!(f32, Avx2Fma, generic_cosine, features = "avx2", "fma");
    export_distance_op!(f32, Avx2Fma, generic_dot, features = "avx2", "fma");
    export_distance_op!(
        f32,
        Avx2Fma,
        generic_squared_euclidean,
        features = "avx2",
        "fma"
    );
    export_vector_x_value_op!(f32, Avx2Fma, generic_div_value, features = "avx2", "fma");

    export_distance_op!(f64, Avx2Fma, generic_cosine, features = "avx2", "fma");
    export_distance_op!(f64, Avx2Fma, generic_dot, features = "avx2", "fma");
    export_distance_op!(
        f64,
        Avx2Fma,
        generic_squared_euclidean,
        features = "avx2",
        "fma"
    );
}

#[cfg(target_arch = "aarch64")]
pub mod neon_ops {
    use super::*;

    export_dense_op!(f32, Neon, mul_dense, features = "neon");
    export_dense_op!(f32, Neon, div_dense, features = "neon");
    export_dense_op!(f32, Neon, add_dense, features = "neon");
    export_dense_op!(f32, Neon, sub_dense, features = "neon");
    export_dense_op!(f32, Neon, max_dense, features = "neon");
    export_dense_op!(f32, Neon, min_dense, features = "neon");

    export_dense_op!(f64, Neon, mul_dense, features = "neon");
    export_dense_op!(f64, Neon, div_dense, features = "neon");
    export_dense_op!(f64, Neon, add_dense, features = "neon");
    export_dense_op!(f64, Neon, sub_dense, features = "neon");
    export_dense_op!(f64, Neon, max_dense, features = "neon");
    export_dense_op!(f64, Neon, min_dense, features = "neon");

    export_distance_op!(f32, Neon, generic_cosine, features = "neon");
    export_distance_op!(f32, Neon, generic_dot, features = "neon");
    export_distance_op!(f32, Neon, generic_squared_euclidean, features = "neon");
    export_vector_x_vector_op!(f32, Neon, generic_add_vector, features = "neon");
    export_vector_x_vector_op!(f32, Neon, generic_sub_vector, features = "neon");
    export_vector_x_vector_op!(f32, Neon, generic_mul_vector, features = "neon");
    export_vector_x_vector_op!(f32, Neon, generic_div_vector, features = "neon");
    export_vector_x_value_op!(f32, Neon, generic_add_value, features = "neon");
    export_vector_x_value_op!(f32, Neon, generic_sub_value, features = "neon");
    export_vector_x_value_op!(f32, Neon, generic_mul_value, features = "neon");
    export_vector_x_value_op!(f32, Neon, generic_div_value, features = "neon");

    export_distance_op!(f64, Neon, generic_cosine, features = "neon");
    export_distance_op!(f64, Neon, generic_dot, features = "neon");
    export_distance_op!(f64, Neon, generic_squared_euclidean, features = "neon");
    export_vector_x_vector_op!(f64, Neon, generic_add_vector, features = "neon");
    export_vector_x_vector_op!(f64, Neon, generic_sub_vector, features = "neon");
    export_vector_x_vector_op!(f64, Neon, generic_mul_vector, features = "neon");
    export_vector_x_vector_op!(f64, Neon, generic_div_vector, features = "neon");
    export_vector_x_value_op!(f64, Neon, generic_add_value, features = "neon");
    export_vector_x_value_op!(f64, Neon, generic_sub_value, features = "neon");
    export_vector_x_value_op!(f64, Neon, generic_mul_value, features = "neon");
    export_vector_x_value_op!(f64, Neon, generic_div_value, features = "neon");
}

#[inline(never)]
pub fn ndarray_dot(a: &ndarray::Array1<f32>, b: &ndarray::Array1<f32>) -> f32 {
    a.dot(b)
}
