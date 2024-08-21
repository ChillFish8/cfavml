//! Comparison related operations
//!
//! Although some of these operations i.e. (maxtarget_features = "avx512f", "avx512bw", mintarget_features = "avx512f", "avx512bw") are technically aggregate
//! routines, they are grouped with the rest of their cmp operations for simplicity.

use crate::buffer::WriteOnlyBuffer;
use crate::danger::{
    generic_cmp_eq_value,
    generic_cmp_eq_vector,
    generic_cmp_gt_value,
    generic_cmp_gt_vector,
    generic_cmp_gte_value,
    generic_cmp_gte_vector,
    generic_cmp_lt_value,
    generic_cmp_lt_vector,
    generic_cmp_lte_value,
    generic_cmp_lte_vector,
    generic_cmp_max,
    generic_cmp_max_value,
    generic_cmp_max_vector,
    generic_cmp_min,
    generic_cmp_min_value,
    generic_cmp_min_vector,
    generic_cmp_neq_value,
    generic_cmp_neq_vector,
    SimdRegister,
};
use crate::math::{AutoMath, Math};

macro_rules! define_op {
    (
        vector_name = $vector_name:ident,
        vector_op = $vector_op:ident,
        vector_doc = $vector_doc:expr,
        value_name = $value_name:ident,
        value_op = $value_op:ident,
        value_doc = $value_doc:expr,
        $imp:ident $(,)?
        $(target_features = $($feat:expr $(,)?)+)?
    ) => {
        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = include_str!($vector_doc)]
        $(

            #[doc = concat!("- ", $("**`+", $feat, "`** ", )*)]
            #[doc = "CPU features are available at runtime. Running on hardware _without_ this feature available will cause immediate UB."]
        )*
        #[doc = r#"
            - The sizes of `a`, `b` and `result` must also be equal to size `dims` otherwise out of
              bounds access can occur.
        "#]
        pub unsafe fn $vector_name<T, B>(
            dims: usize,
            a: &[T],
            b: &[T],
            result: &mut [B],
        )
        where
            T: Copy,
            crate::danger::$imp: SimdRegister<T>,
            AutoMath: Math<T>,
            for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
        {
            $vector_op::<T, crate::danger::$imp, AutoMath, B>(
                dims,
                a,
                b,
                result,
            )
        }

        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = include_str!($value_doc)]
        $(

            #[doc = concat!("- ", $("**`+", $feat, "`** ", )*)]
            #[doc = "CPU features are available at runtime. Running on hardware _without_ this feature available will cause immediate UB."]
        )*
        #[doc = r#"
            - The sizes of `a` and `result` must also be equal to size `dims` otherwise out of
              bounds access can occur.
        "#]
        pub unsafe fn $value_name<T, B>(
            dims: usize,
            value: T,
            a: &[T],
            result: &mut [B],
        )
        where
            T: Copy,
            crate::danger::$imp: SimdRegister<T>,
            AutoMath: Math<T>,
            for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
        {
            $value_op::<T, crate::danger::$imp, AutoMath, B>(
                dims,
                value,
                a,
                result,
            )
        }
    };
}

macro_rules! define_extra_horizontal_op {
    (
        horizontal_name = $horizontal_name:ident,
        horizontal_op = $horizontal_op:ident,
        horizontal_doc = $horizontal_doc:expr,
        $imp:ident $(,)?
        $(target_features = $($feat:expr $(,)?)+)?
    ) => {
        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = include_str!($horizontal_doc)]
        $(

            #[doc = concat!("- ", $("**`+", $feat, "`** ", )*)]
            #[doc = "CPU features are available at runtime. Running on hardware _without_ this feature available will cause immediate UB."]
        )*
        #[doc = r#"
            - The sizes of `a` must also be equal to size `dims` otherwise out of
              bounds access can occur.
        "#]
        pub unsafe fn $horizontal_name<T>(
            dims: usize,
            a: &[T],
        ) -> T
        where
            T: Copy,
            crate::danger::$imp: SimdRegister<T>,
            AutoMath: Math<T>,
        {
            $horizontal_op::<T, crate::danger::$imp, AutoMath>(
                dims,
                a,
            )
        }
    };
}

// OP-max
define_op!(
    vector_name = generic_fallback_cmp_max_vector,
    vector_op = generic_cmp_max_vector,
    vector_doc = "../export_docs/cmp_max_vector.md",
    value_name = generic_fallback_cmp_max_value,
    value_op = generic_cmp_max_value,
    value_doc = "../export_docs/cmp_max_value.md",
    Fallback,
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_op!(
    vector_name = generic_avx2_cmp_max_vector,
    vector_op = generic_cmp_max_vector,
    vector_doc = "../export_docs/cmp_max_vector.md",
    value_name = generic_avx2_cmp_max_value,
    value_op = generic_cmp_max_value,
    value_doc = "../export_docs/cmp_max_value.md",
    Avx2,
    target_features = "avx2"
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_op!(
    vector_name = generic_avx512_cmp_max_vector,
    vector_op = generic_cmp_max_vector,
    vector_doc = "../export_docs/cmp_max_vector.md",
    value_name = generic_avx512_cmp_max_value,
    value_op = generic_cmp_max_value,
    value_doc = "../export_docs/cmp_max_value.md",
    Avx512,
    target_features = "avx512f",
    "avx512bw"
);
#[cfg(target_arch = "aarch64")]
define_op!(
    vector_name = generic_neon_cmp_max_vector,
    vector_op = generic_cmp_max_vector,
    vector_doc = "../export_docs/cmp_max_vector.md",
    value_name = generic_neon_cmp_max_value,
    value_op = generic_cmp_max_value,
    value_doc = "../export_docs/cmp_max_value.md",
    Neon,
    target_features = "neon"
);

// OP-max-horizontal
define_extra_horizontal_op!(
    horizontal_name = generic_fallback_cmp_max,
    horizontal_op = generic_cmp_max,
    horizontal_doc = "../export_docs/cmp_max_horizontal.md",
    Fallback,
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_extra_horizontal_op!(
    horizontal_name = generic_avx2_cmp_max,
    horizontal_op = generic_cmp_max,
    horizontal_doc = "../export_docs/cmp_max_horizontal.md",
    Avx2,
    target_features = "avx2"
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_extra_horizontal_op!(
    horizontal_name = generic_avx512_cmp_max,
    horizontal_op = generic_cmp_max,
    horizontal_doc = "../export_docs/cmp_max_horizontal.md",
    Avx512,
    target_features = "avx512f",
    "avx512bw"
);
#[cfg(target_arch = "aarch64")]
define_extra_horizontal_op!(
    horizontal_name = generic_neon_cmp_max,
    horizontal_op = generic_cmp_max,
    horizontal_doc = "../export_docs/cmp_max_horizontal.md",
    Neon,
    target_features = "neon"
);

// OP-min
define_op!(
    vector_name = generic_fallback_cmp_min_vector,
    vector_op = generic_cmp_min_vector,
    vector_doc = "../export_docs/cmp_min_vector.md",
    value_name = generic_fallback_cmp_min_value,
    value_op = generic_cmp_min_value,
    value_doc = "../export_docs/cmp_min_value.md",
    Fallback,
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_op!(
    vector_name = generic_avx2_cmp_min_vector,
    vector_op = generic_cmp_min_vector,
    vector_doc = "../export_docs/cmp_min_vector.md",
    value_name = generic_avx2_cmp_min_value,
    value_op = generic_cmp_min_value,
    value_doc = "../export_docs/cmp_min_value.md",
    Avx2,
    target_features = "avx2"
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_op!(
    vector_name = generic_avx512_cmp_min_vector,
    vector_op = generic_cmp_min_vector,
    vector_doc = "../export_docs/cmp_min_vector.md",
    value_name = generic_avx512_cmp_min_value,
    value_op = generic_cmp_min_value,
    value_doc = "../export_docs/cmp_min_value.md",
    Avx512,
    target_features = "avx512f",
    "avx512bw"
);
#[cfg(target_arch = "aarch64")]
define_op!(
    vector_name = generic_neon_cmp_min_vector,
    vector_op = generic_cmp_min_vector,
    vector_doc = "../export_docs/cmp_min_vector.md",
    value_name = generic_neon_cmp_min_value,
    value_op = generic_cmp_min_value,
    value_doc = "../export_docs/cmp_min_value.md",
    Neon,
    target_features = "neon"
);

// OP-min-horizontal
define_extra_horizontal_op!(
    horizontal_name = generic_fallback_cmp_min,
    horizontal_op = generic_cmp_min,
    horizontal_doc = "../export_docs/cmp_min_horizontal.md",
    Fallback,
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_extra_horizontal_op!(
    horizontal_name = generic_avx2_cmp_min,
    horizontal_op = generic_cmp_min,
    horizontal_doc = "../export_docs/cmp_min_horizontal.md",
    Avx2,
    target_features = "avx2"
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_extra_horizontal_op!(
    horizontal_name = generic_avx512_cmp_min,
    horizontal_op = generic_cmp_min,
    horizontal_doc = "../export_docs/cmp_min_horizontal.md",
    Avx512,
    target_features = "avx512f",
    "avx512bw"
);
#[cfg(target_arch = "aarch64")]
define_extra_horizontal_op!(
    horizontal_name = generic_neon_cmp_min,
    horizontal_op = generic_cmp_min,
    horizontal_doc = "../export_docs/cmp_min_horizontal.md",
    Neon,
    target_features = "neon"
);

// OP-eq
define_op!(
    vector_name = generic_fallback_cmp_eq_vector,
    vector_op = generic_cmp_eq_vector,
    vector_doc = "../export_docs/cmp_eq_vector.md",
    value_name = generic_fallback_cmp_eq_value,
    value_op = generic_cmp_eq_value,
    value_doc = "../export_docs/cmp_eq_value.md",
    Fallback,
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_op!(
    vector_name = generic_avx2_cmp_eq_vector,
    vector_op = generic_cmp_eq_vector,
    vector_doc = "../export_docs/cmp_eq_vector.md",
    value_name = generic_avx2_cmp_eq_value,
    value_op = generic_cmp_eq_value,
    value_doc = "../export_docs/cmp_eq_value.md",
    Avx2,
    target_features = "avx2"
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_op!(
    vector_name = generic_avx512_cmp_eq_vector,
    vector_op = generic_cmp_eq_vector,
    vector_doc = "../export_docs/cmp_eq_vector.md",
    value_name = generic_avx512_cmp_eq_value,
    value_op = generic_cmp_eq_value,
    value_doc = "../export_docs/cmp_eq_value.md",
    Avx512,
    target_features = "avx512f",
    "avx512bw"
);
#[cfg(target_arch = "aarch64")]
define_op!(
    vector_name = generic_neon_cmp_eq_vector,
    vector_op = generic_cmp_eq_vector,
    vector_doc = "../export_docs/cmp_eq_vector.md",
    value_name = generic_neon_cmp_eq_value,
    value_op = generic_cmp_eq_value,
    value_doc = "../export_docs/cmp_eq_value.md",
    Neon,
    target_features = "neon"
);

// OP-neq
define_op!(
    vector_name = generic_fallback_cmp_neq_vector,
    vector_op = generic_cmp_neq_vector,
    vector_doc = "../export_docs/cmp_neq_vector.md",
    value_name = generic_fallback_cmp_neq_value,
    value_op = generic_cmp_neq_value,
    value_doc = "../export_docs/cmp_neq_value.md",
    Fallback,
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_op!(
    vector_name = generic_avx2_cmp_neq_vector,
    vector_op = generic_cmp_neq_vector,
    vector_doc = "../export_docs/cmp_neq_vector.md",
    value_name = generic_avx2_cmp_neq_value,
    value_op = generic_cmp_neq_value,
    value_doc = "../export_docs/cmp_neq_value.md",
    Avx2,
    target_features = "avx2"
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_op!(
    vector_name = generic_avx512_cmp_neq_vector,
    vector_op = generic_cmp_neq_vector,
    vector_doc = "../export_docs/cmp_neq_vector.md",
    value_name = generic_avx512_cmp_neq_value,
    value_op = generic_cmp_neq_value,
    value_doc = "../export_docs/cmp_neq_value.md",
    Avx512,
    target_features = "avx512f",
    "avx512bw"
);
#[cfg(target_arch = "aarch64")]
define_op!(
    vector_name = generic_neon_cmp_neq_vector,
    vector_op = generic_cmp_neq_vector,
    vector_doc = "../export_docs/cmp_neq_vector.md",
    value_name = generic_neon_cmp_neq_value,
    value_op = generic_cmp_neq_value,
    value_doc = "../export_docs/cmp_neq_value.md",
    Neon,
    target_features = "neon"
);

// OP-lt
define_op!(
    vector_name = generic_fallback_cmp_lt_vector,
    vector_op = generic_cmp_lt_vector,
    vector_doc = "../export_docs/cmp_lt_vector.md",
    value_name = generic_fallback_cmp_lt_value,
    value_op = generic_cmp_lt_value,
    value_doc = "../export_docs/cmp_lt_value.md",
    Fallback,
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_op!(
    vector_name = generic_avx2_cmp_lt_vector,
    vector_op = generic_cmp_lt_vector,
    vector_doc = "../export_docs/cmp_lt_vector.md",
    value_name = generic_avx2_cmp_lt_value,
    value_op = generic_cmp_lt_value,
    value_doc = "../export_docs/cmp_lt_value.md",
    Avx2,
    target_features = "avx2"
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_op!(
    vector_name = generic_avx512_cmp_lt_vector,
    vector_op = generic_cmp_lt_vector,
    vector_doc = "../export_docs/cmp_lt_vector.md",
    value_name = generic_avx512_cmp_lt_value,
    value_op = generic_cmp_lt_value,
    value_doc = "../export_docs/cmp_lt_value.md",
    Avx512,
    target_features = "avx512f",
    "avx512bw"
);
#[cfg(target_arch = "aarch64")]
define_op!(
    vector_name = generic_neon_cmp_lt_vector,
    vector_op = generic_cmp_lt_vector,
    vector_doc = "../export_docs/cmp_lt_vector.md",
    value_name = generic_neon_cmp_lt_value,
    value_op = generic_cmp_lt_value,
    value_doc = "../export_docs/cmp_lt_value.md",
    Neon,
    target_features = "neon"
);

// OP-lte
define_op!(
    vector_name = generic_fallback_cmp_lte_vector,
    vector_op = generic_cmp_lte_vector,
    vector_doc = "../export_docs/cmp_lte_vector.md",
    value_name = generic_fallback_cmp_lte_value,
    value_op = generic_cmp_lte_value,
    value_doc = "../export_docs/cmp_lte_value.md",
    Fallback,
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_op!(
    vector_name = generic_avx2_cmp_lte_vector,
    vector_op = generic_cmp_lte_vector,
    vector_doc = "../export_docs/cmp_lte_vector.md",
    value_name = generic_avx2_cmp_lte_value,
    value_op = generic_cmp_lte_value,
    value_doc = "../export_docs/cmp_lte_value.md",
    Avx2,
    target_features = "avx2"
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_op!(
    vector_name = generic_avx512_cmp_lte_vector,
    vector_op = generic_cmp_lte_vector,
    vector_doc = "../export_docs/cmp_lte_vector.md",
    value_name = generic_avx512_cmp_lte_value,
    value_op = generic_cmp_lte_value,
    value_doc = "../export_docs/cmp_lte_value.md",
    Avx512,
    target_features = "avx512f",
    "avx512bw"
);
#[cfg(target_arch = "aarch64")]
define_op!(
    vector_name = generic_neon_cmp_lte_vector,
    vector_op = generic_cmp_lte_vector,
    vector_doc = "../export_docs/cmp_lte_vector.md",
    value_name = generic_neon_cmp_lte_value,
    value_op = generic_cmp_lte_value,
    value_doc = "../export_docs/cmp_lte_value.md",
    Neon,
    target_features = "neon"
);

// OP-gt
define_op!(
    vector_name = generic_fallback_cmp_gt_vector,
    vector_op = generic_cmp_gt_vector,
    vector_doc = "../export_docs/cmp_gt_vector.md",
    value_name = generic_fallback_cmp_gt_value,
    value_op = generic_cmp_gt_value,
    value_doc = "../export_docs/cmp_gt_value.md",
    Fallback,
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_op!(
    vector_name = generic_avx2_cmp_gt_vector,
    vector_op = generic_cmp_gt_vector,
    vector_doc = "../export_docs/cmp_gt_vector.md",
    value_name = generic_avx2_cmp_gt_value,
    value_op = generic_cmp_gt_value,
    value_doc = "../export_docs/cmp_gt_value.md",
    Avx2,
    target_features = "avx2"
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_op!(
    vector_name = generic_avx512_cmp_gt_vector,
    vector_op = generic_cmp_gt_vector,
    vector_doc = "../export_docs/cmp_gt_vector.md",
    value_name = generic_avx512_cmp_gt_value,
    value_op = generic_cmp_gt_value,
    value_doc = "../export_docs/cmp_gt_value.md",
    Avx512,
    target_features = "avx512f",
    "avx512bw"
);
#[cfg(target_arch = "aarch64")]
define_op!(
    vector_name = generic_neon_cmp_gt_vector,
    vector_op = generic_cmp_gt_vector,
    vector_doc = "../export_docs/cmp_gt_vector.md",
    value_name = generic_neon_cmp_gt_value,
    value_op = generic_cmp_gt_value,
    value_doc = "../export_docs/cmp_gt_value.md",
    Neon,
    target_features = "neon"
);

// OP-gte
define_op!(
    vector_name = generic_fallback_cmp_gte_vector,
    vector_op = generic_cmp_gte_vector,
    vector_doc = "../export_docs/cmp_gte_vector.md",
    value_name = generic_fallback_cmp_gte_value,
    value_op = generic_cmp_gte_value,
    value_doc = "../export_docs/cmp_gte_value.md",
    Fallback,
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_op!(
    vector_name = generic_avx2_cmp_gte_vector,
    vector_op = generic_cmp_gte_vector,
    vector_doc = "../export_docs/cmp_gte_vector.md",
    value_name = generic_avx2_cmp_gte_value,
    value_op = generic_cmp_gte_value,
    value_doc = "../export_docs/cmp_gte_value.md",
    Avx2,
    target_features = "avx2"
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_op!(
    vector_name = generic_avx512_cmp_gte_vector,
    vector_op = generic_cmp_gte_vector,
    vector_doc = "../export_docs/cmp_gte_vector.md",
    value_name = generic_avx512_cmp_gte_value,
    value_op = generic_cmp_gte_value,
    value_doc = "../export_docs/cmp_gte_value.md",
    Avx512,
    target_features = "avx512f",
    "avx512bw"
);
#[cfg(target_arch = "aarch64")]
define_op!(
    vector_name = generic_neon_cmp_gte_vector,
    vector_op = generic_cmp_gte_vector,
    vector_doc = "../export_docs/cmp_gte_vector.md",
    value_name = generic_neon_cmp_gte_value,
    value_op = generic_cmp_gte_value,
    value_doc = "../export_docs/cmp_gte_value.md",
    Neon,
    target_features = "neon"
);

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! define_inner_test {
        ($variant:ident, op = $op:ident, ty = $t:ident, fold_on = $fold_cb:ident) => {
            paste::paste! {
                #[test]
                fn [< $variant _ $op _horizontal_ $t >]() {
                    let (l1, _) = crate::test_utils::get_sample_vectors::<$t>(533);

                    let result = unsafe { [< $variant _cmp_ $op >](l1.len(), &l1) };

                    let expected = l1.iter()
                        .copied()
                        .fold(AutoMath::$fold_cb(), |a, b| AutoMath::[< cmp_ $op >](a, b));
                    assert_eq!(
                        result,
                        expected,
                        "Routine result does not match expected",
                    );
                }

                #[test]
                fn [< $variant _ $op _value_ $t >]() {
                    let (l1, _) = crate::test_utils::get_sample_vectors::<$t>(533);

                    let mut result = vec![$t::default(); 533];
                    unsafe { [< $variant _cmp_ $op _value >](l1.len(), 2 as $t, &l1, &mut result) };

                    let expected = l1.iter()
                        .copied()
                        .map(|v| AutoMath::[< cmp_ $op >](v, 2 as $t))
                        .collect::<Vec<_>>();
                    assert_eq!(
                        result,
                        expected,
                        "Routine result does not match expected",
                    );
                }

                #[test]
                fn [< $variant _ $op _vector_ $t >]() {
                    let (l1, l2) = crate::test_utils::get_sample_vectors::<$t>(533);

                    let mut result = vec![$t::default(); 533];
                    unsafe { [< $variant _cmp_ $op _vector >](l1.len(), &l1, &l2, &mut result) };

                    let expected = l1.iter()
                        .copied()
                        .zip(l2.iter().copied())
                        .map(|(a, b)| AutoMath::[< cmp_ $op >](a, b))
                        .collect::<Vec<_>>();
                    assert_eq!(
                        result,
                        expected,
                        "Routine result does not match expected",
                    );
                }
            }
        };
    }

    macro_rules! define_cmp_test {
        ($variant:ident, types = $($t:ident $(,)?)+) => {
            $(
                define_inner_test!($variant, op = min, ty = $t, fold_on = max);
                define_inner_test!($variant, op = max, ty = $t, fold_on = min);
            )*
        };
    }

    define_cmp_test!(
        generic_fallback,
        types = f32,
        f64,
        i8,
        i16,
        i32,
        i64,
        u8,
        u16,
        u32,
        u64
    );
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    ))]
    define_cmp_test!(
        generic_avx2,
        types = f32,
        f64,
        i8,
        i16,
        i32,
        i64,
        u8,
        u16,
        u32,
        u64
    );
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "nightly",
        target_feature = "avx512f"
    ))]
    define_cmp_test!(
        generic_avx512,
        types = f32,
        f64,
        i8,
        i16,
        i32,
        i64,
        u8,
        u16,
        u32,
        u64
    );
    #[cfg(target_arch = "aarch64")]
    define_cmp_test!(
        generic_neon,
        types = f32,
        f64,
        i8,
        i16,
        i32,
        i64,
        u8,
        u16,
        u32,
        u64
    );
}
