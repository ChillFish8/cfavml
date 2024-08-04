#[macro_export]
/// Dispatches a set of functions based on the available CPU features.
///
/// If the crate is compiled for no-std, this will use compile time dispatch specified by the
/// `target_features` flags, otherwise runtime selection is used.
///
/// Priority is given in the following order:
///
/// #### x86
///
/// - AVX512
/// - AVX2 + FMA
/// - AVX2
/// - Fallback
///
/// #### ARM
///
/// - NEON
/// - Fallback
///
/// ### Usage
///
/// ```
/// use cfavml::dispatch;
///
/// let a = 123;
/// let b = 123;
///
/// dispatch!(
///     avx2fma = my_fma_function => (a, b)
///     avx2 = my_avx2_function => (a, b)
///     neon = my_neon_function => (a, b)
///     fallback = my_fallback_function => (a, b)  // Required!
/// );
///
/// fn my_avx2_function(a: usize, b: usize) {}
/// fn my_neon_function(a: usize, b: usize) {}
/// fn my_fma_function(a: usize, b: usize) {}
/// fn my_fallback_function(a: usize, b: usize) {}
/// ```
///
macro_rules! dispatch {
    (
        $(avx512 = $avx512_fn:expr => ( $($arg1:expr $(,)?)* ) )?
        $(avx2fma = $avx2fma_fn:expr =>( $($arg2:expr $(,)?)* ) )?
        $(avx2 = $avx2_fn:expr => ( $($arg3:expr $(,)?)* ) )?
        $(neon = $neon_fn:expr => ( $($arg4:expr $(,)?)* ) )?
        fallback = $fallback_fn:expr => ( $($arg5:expr $(,)?)* )
    ) => {{
        $(
            #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
            if $crate::dispatch::is_avx512_available() {
                return $avx512_fn($($arg1, )*);
            }
        )?

        $(
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if $crate::dispatch::is_avx2_available() && $crate::dispatch::is_fma_available() {
                return $avx2fma_fn($($arg2, )*);
            }
        )?

        $(
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if $crate::dispatch::is_avx2_available() {
                return $avx2_fn($($arg3, )*);
            }
        )?

        $(
            #[cfg(target_arch = "aarch64")]
            if $crate::dispatch::is_neon_available() {
                return $neon_fn($($arg4, )*);
            }
        )?

        $fallback_fn($($arg5, )*)
    }};
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
#[inline(always)]
/// Returns if AVX512 is available to the system.
///
/// If this is compiling for a no std target, this selection is done
/// at compile time only.
pub fn is_avx512_available() -> bool {
    if cfg!(target_feature = "avx512f") {
        return true;
    }

    #[cfg(feature = "std")]
    if std::arch::is_x86_feature_detected!("avx512f") {
        return true;
    }

    false
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
/// Returns if AVX2 is available to the system.
///
/// If this is compiling for a no std target, this selection is done
/// at compile time only.
pub fn is_avx2_available() -> bool {
    if cfg!(target_feature = "avx2") {
        return true;
    }

    #[cfg(feature = "std")]
    if std::arch::is_x86_feature_detected!("avx2") {
        return true;
    }

    false
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
/// Returns if FMA is available to the system.
///
/// If this is compiling for a no std target, this selection is done
/// at compile time only.
pub fn is_fma_available() -> bool {
    if cfg!(target_feature = "fma") {
        return true;
    }

    #[cfg(feature = "std")]
    if std::arch::is_x86_feature_detected!("fma") {
        return true;
    }

    false
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
/// Returns if NEON is available to the system.
///
/// If this is compiling for a no std target, this selection is done
/// at compile time only.
pub fn is_neon_available() -> bool {
    if cfg!(target_feature = "neon") {
        return true;
    }

    #[cfg(feature = "std")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        return true;
    }

    false
}
