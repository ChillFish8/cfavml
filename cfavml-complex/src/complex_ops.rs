use num_complex::Complex;

pub trait ComplexOps<T>
where
    Complex<T>: Copy,
{
    type Register: Copy;
    //type HalfRegister: Copy;

    /// Extracts the real component from a register
    unsafe fn real_values(value: Self::Register) -> Self::Register;
    /// Extracts the imaginary component from a register
    unsafe fn imag_values(value: Self::Register) -> Self::Register;

    #[inline(always)]
    /// Duplicates the real and imaginary components of a the input into separate registers
    unsafe fn duplicate_complex_components(
        value: Self::Register,
    ) -> (Self::Register, Self::Register) {
        (Self::real_values(value), Self::imag_values(value))
    }

    /// Swaps the real and imaginary components of a complex number
    unsafe fn swap_complex_components(value: Self::Register) -> Self::Register;
}
