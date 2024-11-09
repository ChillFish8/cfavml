use cfavml::danger::SimdRegister;
use num_complex::Complex;

pub trait ComplexOps<T>
where
    Complex<T>: Copy,
{
    type Register: Copy;
    type HalfRegister: Copy;

    /// returns the conjugates of the complex register
    unsafe fn conj(value: Self::Register) -> Self::Register;
    /// returns the inverse of the complex number, adapted from num_complex finv
    unsafe fn inv(value: Self::Register) -> Self::Register;
}

// the ops that only make sense for SIMD registers
pub trait ComplexSimdOps<T>
where
    Complex<T>: Copy,
{
    type Register: Copy;
    /// returns the real components of the complex register duplicated over their imaginary counterparts
    unsafe fn dup_real_components(value: Self::Register) -> Self::Register;
    /// returns the imaginary component of a complex register duplicated over their real counterparts
    unsafe fn dup_imag_components(value: Self::Register) -> Self::Register;
    #[inline(always)]
    /// Duplicates the real and imaginary components of a the input into separate registers
    unsafe fn dup_complex_components(
        value: Self::Register,
    ) -> (Self::Register, Self::Register) {
        (
            Self::dup_real_components(value),
            Self::dup_imag_components(value),
        )
    }
    /// returns the squared_norm of the complex number duplicated into even/odd index pairs
    unsafe fn dup_norm(value: Self::Register) -> Self::Register;

    /// Swaps the real and imaginary components of a complex register
    unsafe fn swap_complex_components(value: Self::Register) -> Self::Register;
}
