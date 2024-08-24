//! Comparison related operations
//!
//! Although some of these operations i.e. (max, min) are technically aggregate
//! routines, they are grouped with the rest of their cmp operations for simplicity.

use crate::buffer::WriteOnlyBuffer;
use crate::danger::{
    generic_cmp_max,
    generic_cmp_min,
    generic_cmp_max_vertical,
    generic_cmp_min_vertical,
    generic_cmp_eq_vertical,
    generic_cmp_neq_vertical,
    generic_cmp_lt_vertical,
    generic_cmp_lte_vertical,
    generic_cmp_gt_vertical,
    generic_cmp_gte_vertical,
    SimdRegister,
};
use crate::math::{AutoMath, Math};
use crate::mem_loader::{MemLoader, IntoMemLoader};

macro_rules! define_op {
    (
        name = $name:ident,
        op = $op:ident,
        doc = $doc:expr,
        $imp:ident $(,)?
        $(target_features = $($feat:expr $(,)?)+)?
    ) => {
        #[inline]
        $(#[target_feature($(enable = $feat, )*)])*
        #[doc = include_str!($doc)]
        $(

            #[doc = concat!("- ", $("**`+", $feat, "`** ", )*)]
            #[doc = "CPU features are available at runtime. Running on hardware _without_ this feature available will cause immediate UB."]
        )*
        pub unsafe fn $name<T, B1, B2, B3>(
            a: B1,
            b: B2,
            result: &mut [B3],
        )
        where
            T: Copy,
            B1: IntoMemLoader<T>,
            B1::Loader: MemLoader<Value = T>,
            B2: IntoMemLoader<T>,
            B2::Loader: MemLoader<Value = T>,
            crate::danger::$imp: SimdRegister<T>,
            AutoMath: Math<T>,
            for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = T>,
        {
            $op::<T, crate::danger::$imp, AutoMath, B1, B2, B3>(
                a,
                b,
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
        pub unsafe fn $horizontal_name<T, B1>(
            a: B1,
        ) -> T
        where
            T: Copy,
            B1: IntoMemLoader<T>,
            B1::Loader: MemLoader<Value = T>,
            crate::danger::$imp: SimdRegister<T>,
            AutoMath: Math<T>,
        {
            $horizontal_op::<T, crate::danger::$imp, AutoMath, B1>(a)
        }
    };
}

// OP-max
define_op!(
    name = generic_fallback_cmp_max_vertical,
    op = generic_cmp_max_vertical,
    doc = "../export_docs/cmp_max_vertical.md",
    Fallback,
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_op!(
    name = generic_avx2_cmp_max_vertical,
    op = generic_cmp_max_vertical,
    doc = "../export_docs/cmp_max_vertical.md",
    Avx2,
    target_features = "avx2"
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_op!(
    name = generic_avx512_cmp_max_vertical,
    op = generic_cmp_max_vertical,
    doc = "../export_docs/cmp_max_vertical.md",
    Avx512,
    target_features = "avx512f",
    "avx512bw"
);
#[cfg(target_arch = "aarch64")]
define_op!(
    name = generic_neon_cmp_max_vertical,
    op = generic_cmp_max_vertical,
    doc = "../export_docs/cmp_max_vertical.md",
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
    name = generic_fallback_cmp_min_vertical,
    op = generic_cmp_min_vertical,
    doc = "../export_docs/cmp_min_vertical.md",
    Fallback,
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_op!(
    name = generic_avx2_cmp_min_vertical,
    op = generic_cmp_min_vertical,
    doc = "../export_docs/cmp_min_vertical.md",
    Avx2,
    target_features = "avx2"
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_op!(
    name = generic_avx512_cmp_min_vertical,
    op = generic_cmp_min_vertical,
    doc = "../export_docs/cmp_min_vertical.md",
    Avx512,
    target_features = "avx512f",
    "avx512bw"
);
#[cfg(target_arch = "aarch64")]
define_op!(
    name = generic_neon_cmp_min_vertical,
    op = generic_cmp_min_vertical,
    doc = "../export_docs/cmp_min_vertical.md",
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
    name = generic_fallback_cmp_eq_vertical,
    op = generic_cmp_eq_vertical,
    doc = "../export_docs/cmp_eq_vertical.md",
    Fallback,
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_op!(
    name = generic_avx2_cmp_eq_vertical,
    op = generic_cmp_eq_vertical,
    doc = "../export_docs/cmp_eq_vertical.md",
    Avx2,
    target_features = "avx2"
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_op!(
    name = generic_avx512_cmp_eq_vertical,
    op = generic_cmp_eq_vertical,
    doc = "../export_docs/cmp_eq_vertical.md",
    Avx512,
    target_features = "avx512f",
    "avx512bw"
);
#[cfg(target_arch = "aarch64")]
define_op!(
    name = generic_neon_cmp_eq_vertical,
    op = generic_cmp_eq_vertical,
    doc = "../export_docs/cmp_eq_vertical.md",
    Neon,
    target_features = "neon"
);

// OP-neq
define_op!(
    name = generic_fallback_cmp_neq_vertical,
    op = generic_cmp_neq_vertical,
    doc = "../export_docs/cmp_neq_vertical.md",
    Fallback,
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_op!(
    name = generic_avx2_cmp_neq_vertical,
    op = generic_cmp_neq_vertical,
    doc = "../export_docs/cmp_neq_vertical.md",
    Avx2,
    target_features = "avx2"
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_op!(
    name = generic_avx512_cmp_neq_vertical,
    op = generic_cmp_neq_vertical,
    doc = "../export_docs/cmp_neq_vertical.md",
    Avx512,
    target_features = "avx512f",
    "avx512bw"
);
#[cfg(target_arch = "aarch64")]
define_op!(
    name = generic_neon_cmp_neq_vertical,
    op = generic_cmp_neq_vertical,
    doc = "../export_docs/cmp_neq_vertical.md",
    Neon,
    target_features = "neon"
);

// OP-lt
define_op!(
    name = generic_fallback_cmp_lt_vertical,
    op = generic_cmp_lt_vertical,
    doc = "../export_docs/cmp_lt_vertical.md",
    Fallback,
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_op!(
    name = generic_avx2_cmp_lt_vertical,
    op = generic_cmp_lt_vertical,
    doc = "../export_docs/cmp_lt_vertical.md",
    Avx2,
    target_features = "avx2"
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_op!(
    name = generic_avx512_cmp_lt_vertical,
    op = generic_cmp_lt_vertical,
    doc = "../export_docs/cmp_lt_vertical.md",
    Avx512,
    target_features = "avx512f",
    "avx512bw"
);
#[cfg(target_arch = "aarch64")]
define_op!(
    name = generic_neon_cmp_lt_vertical,
    op = generic_cmp_lt_vertical,
    doc = "../export_docs/cmp_lt_vertical.md",
    Neon,
    target_features = "neon"
);

// OP-lte
define_op!(
    name = generic_fallback_cmp_lte_vertical,
    op = generic_cmp_lte_vertical,
    doc = "../export_docs/cmp_lte_vertical.md",
    Fallback,
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_op!(
    name = generic_avx2_cmp_lte_vertical,
    op = generic_cmp_lte_vertical,
    doc = "../export_docs/cmp_lte_vertical.md",
    Avx2,
    target_features = "avx2"
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_op!(
    name = generic_avx512_cmp_lte_vertical,
    op = generic_cmp_lte_vertical,
    doc = "../export_docs/cmp_lte_vertical.md",
    Avx512,
    target_features = "avx512f",
    "avx512bw"
);
#[cfg(target_arch = "aarch64")]
define_op!(
    name = generic_neon_cmp_lte_vertical,
    op = generic_cmp_lte_vertical,
    doc = "../export_docs/cmp_lte_vertical.md",
    Neon,
    target_features = "neon"
);

// OP-gt
define_op!(
    name = generic_fallback_cmp_gt_vertical,
    op = generic_cmp_gt_vertical,
    doc = "../export_docs/cmp_gt_vertical.md",
    Fallback,
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_op!(
    name = generic_avx2_cmp_gt_vertical,
    op = generic_cmp_gt_vertical,
    doc = "../export_docs/cmp_gt_vertical.md",
    Avx2,
    target_features = "avx2"
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_op!(
    name = generic_avx512_cmp_gt_vertical,
    op = generic_cmp_gt_vertical,
    doc = "../export_docs/cmp_gt_vertical.md",
    Avx512,
    target_features = "avx512f",
    "avx512bw"
);
#[cfg(target_arch = "aarch64")]
define_op!(
    name = generic_neon_cmp_gt_vertical,
    op = generic_cmp_gt_vertical,
    doc = "../export_docs/cmp_gt_vertical.md",
    Neon,
    target_features = "neon"
);

// OP-gte
define_op!(
    name = generic_fallback_cmp_gte_vertical,
    op = generic_cmp_gte_vertical,
    doc = "../export_docs/cmp_gte_vertical.md",
    Fallback,
);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
define_op!(
    name = generic_avx2_cmp_gte_vertical,
    op = generic_cmp_gte_vertical,
    doc = "../export_docs/cmp_gte_vertical.md",
    Avx2,
    target_features = "avx2"
);
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
define_op!(
    name = generic_avx512_cmp_gte_vertical,
    op = generic_cmp_gte_vertical,
    doc = "../export_docs/cmp_gte_vertical.md",
    Avx512,
    target_features = "avx512f",
    "avx512bw"
);
#[cfg(target_arch = "aarch64")]
define_op!(
    name = generic_neon_cmp_gte_vertical,
    op = generic_cmp_gte_vertical,
    doc = "../export_docs/cmp_gte_vertical.md",
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

                    let result = unsafe { [< $variant _cmp_ $op >](&l1) };

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
                    unsafe { [< $variant _cmp_ $op _vertical >](2 as $t, &l1, &mut result) };

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
                    unsafe { [< $variant _cmp_ $op _vertical >](&l1, &l2, &mut result) };

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
