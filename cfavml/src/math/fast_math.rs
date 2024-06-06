use core::intrinsics;

use super::Math;

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
        a.sqrt()
    }

    #[inline(always)]
    fn abs(a: f32) -> f32 {
        a.abs()
    }

    #[inline(always)]
    fn cmp_eq(a: f32, b: f32) -> bool {
        a == b
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
        a.sqrt()
    }

    #[inline(always)]
    fn abs(a: f64) -> f64 {
        a.abs()
    }

    #[inline(always)]
    fn cmp_eq(a: f64, b: f64) -> bool {
        a == b
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
}
