use super::Math;

/// Standard math operations that apply no specialised handling.
pub struct StdMath;

impl Math<f32> for StdMath {
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
        #[cfg(feature = "std")]
        {
            f32::sqrt(a)
        }

        #[cfg(not(feature = "std"))]
        {
            f32_sqrt_fast(a)
        }
    }

    #[inline(always)]
    fn abs(a: f32) -> f32 {
        #[cfg(feature = "std")]
        {
            f32::abs(a)
        }

        #[cfg(not(feature = "std"))]
        {
            f32_abs_fast(a)
        }
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
        a + b
    }

    #[inline(always)]
    fn sub(a: f32, b: f32) -> f32 {
        a - b
    }

    #[inline(always)]
    fn mul(a: f32, b: f32) -> f32 {
        a * b
    }

    #[inline(always)]
    fn div(a: f32, b: f32) -> f32 {
        a / b
    }

    #[cfg(test)]
    fn is_close(a: f32, b: f32) -> bool {
        let max = a.max(b);
        let min = a.min(b);
        let diff = max - min;
        diff <= 0.00015
    }
}

impl Math<f64> for StdMath {
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
        #[cfg(feature = "std")]
        {
            f64::sqrt(a)
        }

        #[cfg(not(feature = "std"))]
        {
            f32_sqrt_fast(a as f32) as f64
        }
    }

    #[inline(always)]
    fn abs(a: f64) -> f64 {
        #[cfg(feature = "std")]
        {
            f64::abs(a)
        }

        #[cfg(not(feature = "std"))]
        {
            f32_abs_fast(a as f32) as f64
        }
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
        a + b
    }

    #[inline(always)]
    fn sub(a: f64, b: f64) -> f64 {
        a - b
    }

    #[inline(always)]
    fn mul(a: f64, b: f64) -> f64 {
        a * b
    }

    #[inline(always)]
    fn div(a: f64, b: f64) -> f64 {
        a / b
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
        impl Math<$t> for StdMath {
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
                StdMath::sqrt(a as f64) as $t
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
        impl Math<$t> for StdMath {
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
                StdMath::sqrt(a as f64) as $t
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

#[allow(unused)]
#[inline(always)]
/// An approximate f32 sqrt, average deviation of ~5%.
///
/// This is an _approximate_ function, it is faster, but primarily designed
/// to just be used for the no_std target since we cannot use the inbuilt methods.
fn f32_sqrt_fast(a: f32) -> f32 {
    if a >= 0.0 {
        f32::from_bits((a.to_bits() + 0x3f80_0000) >> 1)
    } else {
        f32::NAN
    }
}

#[allow(unused)]
#[inline(always)]
/// Computes the ABS of a f32.
fn f32_abs_fast(a: f32) -> f32 {
    const SIGN_MASK: u32 = 0b1000_0000_0000_0000_0000_0000_0000_0000;
    f32::from_bits(a.to_bits() & !SIGN_MASK)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sqrt_sanity() {
        let a = f32::sqrt(1.234294);
        let b = f32_sqrt_fast(1.23429);
        assert_eq!(a, 1.1109879);
        assert_eq!(b, 1.117145);
    }
}
