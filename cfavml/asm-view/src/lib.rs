#![allow(clippy::missing_safety_doc)]

macro_rules! define_distance_exports {
    ($op:ident) => {
        paste::paste! {
            #[no_mangle]
            #[inline(never)]
            pub unsafe fn [< xany_avx2_nofma_ $op >] (x: &[f32], y: &[f32]) -> f32 {
                eonn_accel::danger::[< f32_xany_avx2_nofma_ $op >](x, y)
            }

            #[no_mangle]
            #[inline(never)]
            pub unsafe fn [< xconst_avx2_nofma_  $op >](x: &[f32], y: &[f32]) -> f32 {
                eonn_accel::danger::[< f32_xconst_avx2_nofma_  $op >]::<1024>(x, y)
            }

            #[no_mangle]
            #[inline(never)]
            pub unsafe fn [< xany_avx2_fma_ $op >](x: &[f32], y: &[f32]) -> f32 {
                eonn_accel::danger::[< f32_xany_avx2_fma_ $op >](x, y)
            }

            #[no_mangle]
            #[inline(never)]
            pub unsafe fn [< xconst_avx2_fma_ $op >](x: &[f32], y: &[f32]) -> f32 {
                eonn_accel::danger::[< f32_xconst_avx2_fma_ $op >]::<1024>(x, y)
            }

            #[no_mangle]
            #[inline(never)]
            pub unsafe fn [< xany_avx512_fma_ $op >](x: &[f32], y: &[f32]) -> f32 {
                eonn_accel::danger::[< f32_xany_avx512_fma_ $op >](x, y)
            }

            #[no_mangle]
            #[inline(never)]
            pub unsafe fn [< xconst_avx512_fma_ $op >](x: &[f32], y: &[f32]) -> f32 {
                eonn_accel::danger::[< f32_xconst_avx512_fma_ $op >]::<1024>(x, y)
            }

            #[no_mangle]
            #[inline(never)]
            pub unsafe fn [< xany_fallback_nofma_ $op >](x: &[f32], y: &[f32]) -> f32 {
                eonn_accel::danger::[< f32_xany_fallback_nofma_ $op >](x, y)
            }
        }
    };
}

define_distance_exports!(dot);
define_distance_exports!(cosine);
define_distance_exports!(euclidean);
