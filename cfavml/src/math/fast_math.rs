use core::intrinsics;

use super::{Math, StdMath};

/// Basic math operations backed by fast-math intrinsics.
pub struct FastMath;

impl Math<f32> for FastMath {
    #[inline(always)]
    fn zero() -> f32 {
        0.0
    }

    #[inline(always)]
    fn one() -> f32 {
        1.0
    }

    #[inline(always)]
    fn max() -> f32 {
        f32::INFINITY
    }

    #[inline(always)]
    fn min() -> f32 {
        f32::NEG_INFINITY
    }

    #[inline(always)]
    fn sqrt(a: f32) -> f32 {
        core::intrinsics::sqrtf32(a)
    }

    #[inline(always)]
    fn abs(a: f32) -> f32 {
        StdMath::abs(a)
    }

    #[inline(always)]
    fn cmp_eq(a: f32, b: f32) -> bool {
        a == b
    }

    #[inline(always)]
    fn cmp_lt(a: f32, b: f32) -> bool {
        a < b
    }

    #[inline(always)]
    fn cmp_lte(a: f32, b: f32) -> bool {
        a <= b
    }

    #[inline(always)]
    fn cmp_gt(a: f32, b: f32) -> bool {
        a > b
    }

    #[inline(always)]
    fn cmp_gte(a: f32, b: f32) -> bool {
        a >= b
    }

    #[inline(always)]
    fn cmp_min(a: f32, b: f32) -> f32 {
        a.min(b)
    }

    #[inline(always)]
    fn cmp_max(a: f32, b: f32) -> f32 {
        a.max(b)
    }

    #[inline(always)]
    fn add(a: f32, b: f32) -> f32 {
        if cfg!(miri) {
            a + b
        } else {
            intrinsics::fadd_algebraic(a, b)
        }
    }

    #[inline(always)]
    fn sub(a: f32, b: f32) -> f32 {
        if cfg!(miri) {
            a - b
        } else {
            intrinsics::fsub_algebraic(a, b)
        }
    }

    #[inline(always)]
    fn mul(a: f32, b: f32) -> f32 {
        if cfg!(miri) {
            a * b
        } else {
            intrinsics::fmul_algebraic(a, b)
        }
    }

    #[inline(always)]
    fn div(a: f32, b: f32) -> f32 {
        if cfg!(miri) {
            a / b
        } else {
            intrinsics::fdiv_algebraic(a, b)
        }
    }

    #[cfg(test)]
    fn is_close(a: f32, b: f32) -> bool {
        let max = a.max(b);
        let min = a.min(b);
        let diff = max - min;
        diff <= 0.00015
    }
}

impl Math<f64> for FastMath {
    #[inline(always)]
    fn zero() -> f64 {
        0.0
    }

    #[inline(always)]
    fn one() -> f64 {
        1.0
    }

    #[inline(always)]
    fn max() -> f64 {
        f64::INFINITY
    }

    #[inline(always)]
    fn min() -> f64 {
        f64::NEG_INFINITY
    }

    #[inline(always)]
    fn sqrt(a: f64) -> f64 {
        core::intrinsics::sqrtf64(a)
    }

    #[inline(always)]
    fn abs(a: f64) -> f64 {
        StdMath::abs(a)
    }

    #[inline(always)]
    fn cmp_eq(a: f64, b: f64) -> bool {
        a == b
    }

    #[inline(always)]
    fn cmp_lt(a: f64, b: f64) -> bool {
        a < b
    }

    #[inline(always)]
    fn cmp_lte(a: f64, b: f64) -> bool {
        a <= b
    }

    #[inline(always)]
    fn cmp_gt(a: f64, b: f64) -> bool {
        a > b
    }

    #[inline(always)]
    fn cmp_gte(a: f64, b: f64) -> bool {
        a >= b
    }

    #[inline(always)]
    fn cmp_min(a: f64, b: f64) -> f64 {
        a.min(b)
    }

    #[inline(always)]
    fn cmp_max(a: f64, b: f64) -> f64 {
        a.max(b)
    }

    #[inline(always)]
    fn add(a: f64, b: f64) -> f64 {
        if cfg!(miri) {
            a + b
        } else {
            intrinsics::fadd_algebraic(a, b)
        }
    }

    #[inline(always)]
    fn sub(a: f64, b: f64) -> f64 {
        if cfg!(miri) {
            a - b
        } else {
            intrinsics::fsub_algebraic(a, b)
        }
    }

    #[inline(always)]
    fn mul(a: f64, b: f64) -> f64 {
        if cfg!(miri) {
            a * b
        } else {
            intrinsics::fmul_algebraic(a, b)
        }
    }

    #[inline(always)]
    fn div(a: f64, b: f64) -> f64 {
        if cfg!(miri) {
            a / b
        } else {
            intrinsics::fdiv_algebraic(a, b)
        }
    }

    #[cfg(test)]
    fn is_close(a: f64, b: f64) -> bool {
        let max = a.max(b);
        let min = a.min(b);
        let diff = max - min;
        diff <= 0.00015
    }
}

macro_rules! define_int_ops {
    ($t:ident) => {
        impl Math<$t> for FastMath {
            #[inline(always)]
            fn zero() -> $t {
                0
            }

            #[inline(always)]
            fn one() -> $t {
                1
            }

            #[inline(always)]
            fn max() -> $t {
                $t::MAX
            }

            #[inline(always)]
            fn min() -> $t {
                $t::MIN
            }

            #[inline(always)]
            fn sqrt(a: $t) -> $t {
                FastMath::sqrt(a as f64) as $t
            }

            #[inline(always)]
            fn abs(a: $t) -> $t {
                a.abs()
            }

            #[inline(always)]
            fn cmp_eq(a: $t, b: $t) -> bool {
                a == b
            }

            #[inline(always)]
            fn cmp_lt(a: $t, b: $t) -> bool {
                a < b
            }

            #[inline(always)]
            fn cmp_lte(a: $t, b: $t) -> bool {
                a <= b
            }

            #[inline(always)]
            fn cmp_gt(a: $t, b: $t) -> bool {
                a > b
            }

            #[inline(always)]
            fn cmp_gte(a: $t, b: $t) -> bool {
                a >= b
            }

            #[inline(always)]
            fn cmp_min(a: $t, b: $t) -> $t {
                a.min(b)
            }

            #[inline(always)]
            fn cmp_max(a: $t, b: $t) -> $t {
                a.max(b)
            }

            #[inline(always)]
            fn add(a: $t, b: $t) -> $t {
                a.wrapping_add(b)
            }

            #[inline(always)]
            fn sub(a: $t, b: $t) -> $t {
                a.wrapping_sub(b)
            }

            #[inline(always)]
            fn mul(a: $t, b: $t) -> $t {
                a.wrapping_mul(b)
            }

            #[inline(always)]
            fn div(a: $t, b: $t) -> $t {
                a.wrapping_div(b)
            }

            #[cfg(test)]
            fn is_close(a: $t, b: $t) -> bool {
                a == b
            }
        }
    };
    (unsigned $t:ident) => {
        impl Math<$t> for FastMath {
            #[inline(always)]
            fn zero() -> $t {
                0
            }

            #[inline(always)]
            fn one() -> $t {
                1
            }

            #[inline(always)]
            fn max() -> $t {
                $t::MAX
            }

            #[inline(always)]
            fn min() -> $t {
                $t::MIN
            }

            #[inline(always)]
            fn sqrt(a: $t) -> $t {
                FastMath::sqrt(a as f64) as $t
            }

            #[inline(always)]
            fn abs(a: $t) -> $t {
                a
            }

            #[inline(always)]
            fn cmp_eq(a: $t, b: $t) -> bool {
                a == b
            }

            #[inline(always)]
            fn cmp_lt(a: $t, b: $t) -> bool {
                a < b
            }

            #[inline(always)]
            fn cmp_lte(a: $t, b: $t) -> bool {
                a <= b
            }

            #[inline(always)]
            fn cmp_gt(a: $t, b: $t) -> bool {
                a > b
            }

            #[inline(always)]
            fn cmp_gte(a: $t, b: $t) -> bool {
                a >= b
            }

            #[inline(always)]
            fn cmp_min(a: $t, b: $t) -> $t {
                a.min(b)
            }

            #[inline(always)]
            fn cmp_max(a: $t, b: $t) -> $t {
                a.max(b)
            }

            #[inline(always)]
            fn add(a: $t, b: $t) -> $t {
                a.wrapping_add(b)
            }

            #[inline(always)]
            fn sub(a: $t, b: $t) -> $t {
                a.wrapping_sub(b)
            }

            #[inline(always)]
            fn mul(a: $t, b: $t) -> $t {
                a.wrapping_mul(b)
            }

            #[inline(always)]
            fn div(a: $t, b: $t) -> $t {
                a.wrapping_div(b)
            }

            #[cfg(test)]
            fn is_close(a: $t, b: $t) -> bool {
                a == b
            }
        }
    };
}

define_int_ops!(i8);
define_int_ops!(i16);
define_int_ops!(i32);
define_int_ops!(i64);

define_int_ops!(unsigned u8);
define_int_ops!(unsigned u16);
define_int_ops!(unsigned u32);
define_int_ops!(unsigned u64);
