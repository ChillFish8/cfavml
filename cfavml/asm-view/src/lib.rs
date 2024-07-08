use cfavml::danger::*;

macro_rules! export_dense_op {
    ($t:ident, $im:ident, $op:ident, features = $($feat:expr $(,)?)*) => {
        paste::paste!{
            #[inline(never)]
            #[target_feature($(enable = $feat)*)]
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
            #[target_feature($(enable = $feat)*)]
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
            #[target_feature($(enable = $feat)*)]
            pub unsafe fn [<impl_ $im:lower _ $t _ $op>](a: &[$t], b: &[$t], res: &mut [$t]) {
                let res = $op::<_, $im, cfavml::math::AutoMath>(a.len(), a, b, res)  ;
                std::hint::black_box(res);
            }
        }
    };
}

macro_rules! export_vector_x_value_op {
    ($t:ident, $im:ident, $op:ident, features = $($feat:expr $(,)?)*) => {
        paste::paste!{
            #[inline(never)]
            #[target_feature($(enable = $feat)*)]
            pub unsafe fn [<impl_ $im:lower _ $t _ $op>](value: $t, a: &[$t], res: &mut [$t]) {
                let res = $op::<_, $im, cfavml::math::AutoMath>(a.len(), value, a, res)  ;
                std::hint::black_box(res);
            }
        }
    };
}

export_dense_op!(i8, Avx2, mul_dense, features = "avx2");
export_dense_op!(i8, Avx2, div_dense, features = "avx2");
export_dense_op!(i8, Avx2, add_dense, features = "avx2");
export_dense_op!(i8, Avx2, sub_dense, features = "avx2");
export_dense_op!(i8, Avx2, max_dense, features = "avx2");
export_dense_op!(i8, Avx2, min_dense, features = "avx2");

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
export_distance_op!(i8, Avx2, generic_dot_product, features = "avx2");
export_distance_op!(i8, Avx2, generic_euclidean, features = "avx2");
export_vector_x_vector_op!(i8, Avx2, generic_add_vector, features = "avx2");
export_vector_x_vector_op!(i8, Avx2, generic_sub_vector, features = "avx2");
export_vector_x_vector_op!(i8, Avx2, generic_mul_vector, features = "avx2");
export_vector_x_vector_op!(i8, Avx2, generic_div_vector, features = "avx2");
export_vector_x_value_op!(i8, Avx2, generic_add_value, features = "avx2");
export_vector_x_value_op!(i8, Avx2, generic_sub_value, features = "avx2");
export_vector_x_value_op!(i8, Avx2, generic_mul_value, features = "avx2");
export_vector_x_value_op!(i8, Avx2, generic_div_value, features = "avx2");

export_distance_op!(f32, Avx2, generic_cosine, features = "avx2");
export_distance_op!(f32, Avx2, generic_dot_product, features = "avx2");
export_distance_op!(f32, Avx2, generic_euclidean, features = "avx2");
export_vector_x_vector_op!(f32, Avx2, generic_add_vector, features = "avx2");
export_vector_x_vector_op!(f32, Avx2, generic_sub_vector, features = "avx2");
export_vector_x_vector_op!(f32, Avx2, generic_mul_vector, features = "avx2");
export_vector_x_vector_op!(f32, Avx2, generic_div_vector, features = "avx2");
export_vector_x_value_op!(f32, Avx2, generic_add_value, features = "avx2");
export_vector_x_value_op!(f32, Avx2, generic_sub_value, features = "avx2");
export_vector_x_value_op!(f32, Avx2, generic_mul_value, features = "avx2");
export_vector_x_value_op!(f32, Avx2, generic_div_value, features = "avx2");

export_distance_op!(f64, Avx2, generic_cosine, features = "avx2");
export_distance_op!(f64, Avx2, generic_dot_product, features = "avx2");
export_distance_op!(f64, Avx2, generic_euclidean, features = "avx2");
export_vector_x_vector_op!(f64, Avx2, generic_add_vector, features = "avx2");
export_vector_x_vector_op!(f64, Avx2, generic_sub_vector, features = "avx2");
export_vector_x_vector_op!(f64, Avx2, generic_mul_vector, features = "avx2");
export_vector_x_vector_op!(f64, Avx2, generic_div_vector, features = "avx2");
export_vector_x_value_op!(f64, Avx2, generic_add_value, features = "avx2");
export_vector_x_value_op!(f64, Avx2, generic_sub_value, features = "avx2");
export_vector_x_value_op!(f64, Avx2, generic_mul_value, features = "avx2");
export_vector_x_value_op!(f64, Avx2, generic_div_value, features = "avx2");
