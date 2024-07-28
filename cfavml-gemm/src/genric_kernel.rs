use std::fmt::Debug;
use std::{mem, ptr};

use cfavml::danger::{DenseLane, SimdRegister};
use cfavml_utils::aligned_buffer::AlignedBuffer;

/// A generic matrix micro kernel.
///
/// The reason bits are being used here is because this kernel is generic over
/// the actual type being used, so for `f32` this becomes a 8x8 kernel, `f64` becomes
/// `4x4` etc... Internally these all become operations over a [SimdRegister] with
/// a 8x dense lane.
pub(crate) struct GenericMatrixKernel<T, R>
where
    T: Copy,
    R: SimdRegister<T>,
{
    /// THe number of dimensions of the individual vectors of the matrix.
    dims: usize,
    /// Our row components, primarily targeting to be cached in L2, but this is largely
    /// up to the CPU to do so, each element is loaded individually and broadcast
    /// across registers.
    a_buffer: AlignedBuffer<T>,
    /// Our column components, primarily targeting to be cached in L1, but this is largely
    /// up to the CPU to do so, this buffer is loaded in _rows_ rather than columns.
    b_buffer: AlignedBuffer<T>,
    /// A temporary allocation buffer tha can be used as a scratch space.
    temp_buffer: AlignedBuffer<T>,
    /// Our matmul result for the micro kernel, these are kept within registers.
    matrix_result: DenseLane<R::Register>,
}

impl<T, R> GenericMatrixKernel<T, R>
where
    T: Copy + Debug + 'static,
    R: SimdRegister<T>,
{
    /// Creates a new micro kernel with an allocation large enough
    /// to handle the 256bit x 8 operation.
    pub(crate) fn allocate(dims: usize) -> Self {
        // We select `num_elements_per_lane * 8` rows+cols to compute
        // the dot product with, this means on f32 this is `8 * 8 == 64` elements
        // amounting to `8 * dims` elements loaded from buffers `a` and `b`.
        let num_elements_per_lane = R::elements_per_lane();
        let buffer_size = dims * num_elements_per_lane;

        // SAFETY:
        // Internally this is only ever going to be used with integer primitives
        // which do not impose any additional alignment issues.
        let a_buffer = unsafe { AlignedBuffer::zeroed(buffer_size) };
        let b_buffer = unsafe { AlignedBuffer::zeroed(buffer_size) };
        let temp_buffer = unsafe { AlignedBuffer::zeroed(buffer_size) };

        Self {
            dims,
            a_buffer,
            b_buffer,
            temp_buffer,
            matrix_result: unsafe { R::zeroed_dense() },
        }
    }

    /// Computes the dot product of the given matrix of `a` and `b` writing the result to `result`
    /// using the configured micro-kernel.
    pub(crate) unsafe fn dot_matrix(
        &mut self,
        a_width: usize,
        a_height: usize,
        a: &[T],
        b_width: usize,
        b_height: usize,
        b: &[T],
        result: &mut [T],
    ) {
        assert_eq!(
            a.len(),
            a_width * a_height,
            "Input matrix `a` size does not match dimensions"
        );
        assert_eq!(
            b.len(),
            b_width * b_height,
            "Input matrix `b` size does not match dimensions"
        );
        assert_eq!(
            result.len(),
            a_width * b_height,
            "Output matrix `result` size does not match dimensions"
        );

        assert_eq!(
            a_width % R::elements_per_lane(),
            0,
            "TODO: Remove limitation"
        ); // TODO: FIXME!
        assert_eq!(
            a_height % R::elements_per_lane(),
            0,
            "TODO: Remove limitation"
        ); // TODO: FIXME!
        assert_eq!(
            b_width % R::elements_per_lane(),
            0,
            "TODO: Remove limitation"
        ); // TODO: FIXME!
        assert_eq!(
            b_height % R::elements_per_lane(),
            0,
            "TODO: Remove limitation"
        ); // TODO: FIXME!

        let b_buffer_ptr = self.b_buffer.as_mut_slice().as_mut_ptr();

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let result_ptr = result.as_mut_ptr();

        let mut i = 0;
        while i < a_height {
            // Because we swap buffers during transposition we need to get the new
            // pointer each time.
            let a_buffer_ptr = self.a_buffer.as_mut_slice().as_mut_ptr();

            ptr::copy_nonoverlapping(
                a_ptr.add(i * a_width),
                a_buffer_ptr,
                R::elements_per_lane() * a_width,
            );
            self.prep_buffers();

            let mut j = 0;
            while j < b_width {
                for n in 0..b_height {
                    let reg = R::load(b_ptr.add(j + (n * b_width)));
                    R::write(b_buffer_ptr.add(n * 8), reg);
                }

                self.execute_kernel();
                self.drain_kernel(result_ptr.add(j + (i * a_width)), a_width);

                j += R::elements_per_lane();
            }

            i += R::elements_per_lane();
        }
    }

    /// Does some pre-processing on the buffers so they can be operated on more efficiently.
    unsafe fn prep_buffers(&mut self) {
        crate::transpose::transpose_matrix(
            self.dims,
            R::elements_per_lane(),
            &self.a_buffer,
            self.temp_buffer.as_mut_slice(),
        );

        mem::swap(&mut self.a_buffer, &mut self.temp_buffer);
    }

    #[inline(always)]
    /// Drain the current kernel matrix result registers and store the results
    /// in the provided address.
    unsafe fn drain_kernel(&mut self, result_ptr: *mut T, step_by: usize) {
        R::write(result_ptr.add(0 * step_by), self.matrix_result.a);
        R::write(result_ptr.add(1 * step_by), self.matrix_result.b);
        R::write(result_ptr.add(2 * step_by), self.matrix_result.c);
        R::write(result_ptr.add(3 * step_by), self.matrix_result.d);
        R::write(result_ptr.add(4 * step_by), self.matrix_result.e);
        R::write(result_ptr.add(5 * step_by), self.matrix_result.f);
        R::write(result_ptr.add(6 * step_by), self.matrix_result.g);
        R::write(result_ptr.add(7 * step_by), self.matrix_result.h);
        self.matrix_result = unsafe { R::zeroed_dense() };
    }

    /// Executes the micro-kernel using the configured buffers.
    unsafe fn execute_kernel(&mut self) {
        let a_ptr = self.a_buffer.as_ptr();
        let b_ptr = self.b_buffer.as_ptr();

        // So future me:
        // Here is the plan of the first generation because I can't remember how the BLIS papers
        // did it...
        //
        // - We do an initial naive test loading a and broadcasting it across registers.
        // - Then we go over the _full_ row in B and do the dot product intermediate result for
        //   that row.
        // - Then we advance `a` by 1, load, broadcast, repeat.
        // - No idea how this is gonna go, maybe it is worth transposing `a` to be column-major?
        //   * Reason I say this is because it is a bit of a waste to go through `b` completely
        //     for just one element of `a`, since that ends up being `dims` iterations per element
        //     of `a` and loading the memory each time!
        //   * Maybe transposing to column-major is too expensive here though. - EDIT: It isn't
        // - Update: we've transposed the sub-matrix of `a` so it is column major.
        // - TODO: We should handle width and heights of the two matrices separately
        //         otherwise it is going to become painful later on.
        // - TODO: So it definitely does not work when the matrix is odd lengths, i.e. 32 x 8.

        let mut offset = 0;
        while offset < self.a_buffer.len() {
            self.compute_partial_step(a_ptr.add(offset), b_ptr.add(offset));
            offset += 16;
        }
    }

    #[inline(always)]
    /// This block calculates 2 out of `N` dimensions of the matrix rows/columns, adding
    /// the intermediate result to the result matrix. We try and maximise the instruction
    /// throughput here although we only target ~16 registers in use at any point in time
    /// which may not be fully optimal on AVX512 or NEON architectures which have 32 available.
    unsafe fn compute_partial_step(&mut self, a_ptr: *const T, b_ptr: *const T) {
        let b_row_1 = R::load(b_ptr);
        let b_row_2 = R::load(b_ptr.add(8));

        // Processing the first row of the first 4 columns of B
        // and then calculating the first part of the dot product of the first element of
        // the first 4 rows  of the 1st column.
        let a_broadcast_col1_row1 = R::filled(a_ptr.read());
        let a_broadcast_col1_row2 = R::filled(a_ptr.add(1).read());
        let a_broadcast_col1_row3 = R::filled(a_ptr.add(2).read());
        let a_broadcast_col1_row4 = R::filled(a_ptr.add(3).read());
        self.matrix_result.a =
            R::fmadd(a_broadcast_col1_row1, b_row_1, self.matrix_result.a);
        self.matrix_result.b =
            R::fmadd(a_broadcast_col1_row2, b_row_1, self.matrix_result.b);
        self.matrix_result.c =
            R::fmadd(a_broadcast_col1_row3, b_row_1, self.matrix_result.c);
        self.matrix_result.d =
            R::fmadd(a_broadcast_col1_row4, b_row_1, self.matrix_result.d);

        // Calculating the second step, forming the first step of calculating the 8x8 matrix.
        let a_broadcast_col1_row5 = R::filled(a_ptr.add(4).read());
        let a_broadcast_col1_row6 = R::filled(a_ptr.add(5).read());
        let a_broadcast_col1_row7 = R::filled(a_ptr.add(6).read());
        let a_broadcast_col1_row8 = R::filled(a_ptr.add(7).read());
        self.matrix_result.e =
            R::fmadd(a_broadcast_col1_row5, b_row_1, self.matrix_result.e);
        self.matrix_result.f =
            R::fmadd(a_broadcast_col1_row6, b_row_1, self.matrix_result.f);
        self.matrix_result.g =
            R::fmadd(a_broadcast_col1_row7, b_row_1, self.matrix_result.g);
        self.matrix_result.h =
            R::fmadd(a_broadcast_col1_row8, b_row_1, self.matrix_result.h);

        // Now we've gone down to the 2nd row of B, and target the 2nd col of A.
        let a_broadcast_col2_row1 = R::filled(a_ptr.add(8).read());
        let a_broadcast_col2_row2 = R::filled(a_ptr.add(9).read());
        let a_broadcast_col2_row3 = R::filled(a_ptr.add(10).read());
        let a_broadcast_col2_row4 = R::filled(a_ptr.add(11).read());
        self.matrix_result.a =
            R::fmadd(a_broadcast_col2_row1, b_row_2, self.matrix_result.a);
        self.matrix_result.b =
            R::fmadd(a_broadcast_col2_row2, b_row_2, self.matrix_result.b);
        self.matrix_result.c =
            R::fmadd(a_broadcast_col2_row3, b_row_2, self.matrix_result.c);
        self.matrix_result.d =
            R::fmadd(a_broadcast_col2_row4, b_row_2, self.matrix_result.d);

        let a_broadcast_col2_row5 = R::filled(a_ptr.add(12).read());
        let a_broadcast_col2_row6 = R::filled(a_ptr.add(13).read());
        let a_broadcast_col2_row7 = R::filled(a_ptr.add(14).read());
        let a_broadcast_col2_row8 = R::filled(a_ptr.add(15).read());
        self.matrix_result.e =
            R::fmadd(a_broadcast_col2_row5, b_row_2, self.matrix_result.e);
        self.matrix_result.f =
            R::fmadd(a_broadcast_col2_row6, b_row_2, self.matrix_result.f);
        self.matrix_result.g =
            R::fmadd(a_broadcast_col2_row7, b_row_2, self.matrix_result.g);
        self.matrix_result.h =
            R::fmadd(a_broadcast_col2_row8, b_row_2, self.matrix_result.h);
    }
}

fn debug_slice<T: Debug>(prefix: &str, step: usize, buffer: &[T]) {
    let mut i = 0;
    while i < buffer.len() {
        let slice = &buffer[i..i + step];
        println!("{prefix}: {slice:?}");
        i += step;
    }
    println!();
}

#[cfg(test)]
#[rustfmt::skip]
mod tests {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use cfavml::danger::Avx2;

    use super::*;

    const SEED: u64 = 2837564324875;

    fn generate_sample_data(size: usize) -> (Vec<f32>, Vec<f32>) {
        let mut rng = ChaCha8Rng::seed_from_u64(SEED);

        let mut a = Vec::with_capacity(size);
        let mut b = Vec::with_capacity(size);

        for _ in 0..size {
            a.push(rng.gen::<u8>() as f32);
            b.push(rng.gen::<u8>() as f32);
        }

        (a, b)
    }

    #[test]
    fn test_basic_f32_gemm_8x8() {
        let (a, b) = generate_sample_data(8 * 8);

        let na = ndarray::ArrayView2::from_shape((8, 8), &a).unwrap();
        let nb = ndarray::ArrayView2::from_shape((8, 8), &b).unwrap();

        let mut result_buffer = vec![0.0; 8 * 8];

        let mut kernel = GenericMatrixKernel::<f32, Avx2>::allocate(8);
        unsafe {
            kernel.dot_matrix(
                8,
                8,
                &a,
                8,
                8,
                &b,
                &mut result_buffer,
            );
        }

        let mut expected = ndarray::Array2::zeros((8, 8));
        ndarray::linalg::general_mat_mul(
            1.0,
            &na,
            &nb,
            0.0,
            &mut expected,
        );

        assert_eq!(&expected.into_raw_vec(), &result_buffer);
    }

    #[test]
    fn test_basic_f32_gemm_16x16() {
        let (a, b) = generate_sample_data(16 * 16);

        // Use this for debugging if the test fails:
        // let a = [
        //     0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0,  11.0,  12.0,  13.0,  14.0,  15.0,
        //     0.1,  1.1,  2.1,  3.1,  4.1,  5.1,  6.1,  7.1,  8.1,  9.1,  10.1,  11.1,  12.1,  13.1,  14.1,  15.1,
        //     0.2,  1.2,  2.2,  3.2,  4.2,  5.2,  6.2,  7.2,  8.2,  9.2,  10.2,  11.2,  12.2,  13.2,  14.2,  15.2,
        //     0.3,  1.3,  2.3,  3.3,  4.3,  5.3,  6.3,  7.3,  8.3,  9.3,  10.3,  11.3,  12.3,  13.3,  14.3,  15.3,
        //     0.4,  1.4,  2.4,  3.4,  4.4,  5.4,  6.4,  7.4,  8.4,  9.4,  10.4,  11.4,  12.4,  13.4,  14.4,  15.4,
        //     0.5,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,  9.5,  10.5,  11.5,  12.5,  13.5,  14.5,  15.5,
        //     0.6,  1.6,  2.6,  3.6,  4.6,  5.6,  6.6,  7.6,  8.6,  9.6,  10.6,  11.6,  12.6,  13.6,  14.6,  15.6,
        //     0.7,  1.7,  2.7,  3.7,  4.7,  5.7,  6.7,  7.7,  8.7,  9.7,  10.7,  11.7,  12.7,  13.7,  14.7,  15.7,
        //     0.8,  1.8,  2.8,  3.8,  4.8,  5.8,  6.8,  7.8,  8.8,  9.8,  10.8,  11.8,  12.8,  13.8,  14.8,  15.8,
        //     0.9,  1.9,  2.9,  3.9,  4.9,  5.9,  6.9,  7.9,  8.9,  9.9,  10.9,  11.9,  12.9,  13.9,  14.9,  15.9,
        //     0.10, 1.10, 2.10, 3.10, 4.10, 5.10, 6.10, 7.10, 8.10, 9.10, 10.10, 11.10, 12.10, 13.10, 14.10, 15.10,
        //     0.11, 1.11, 2.11, 3.11, 4.11, 5.11, 6.11, 7.11, 8.11, 9.11, 10.11, 11.11, 12.11, 13.11, 14.11, 15.11,
        //     0.12, 1.12, 2.12, 3.12, 4.12, 5.12, 6.12, 7.12, 8.12, 9.12, 10.12, 11.12, 12.12, 13.12, 14.12, 15.12,
        //     0.13, 1.13, 2.13, 3.13, 4.13, 5.13, 6.13, 7.13, 8.13, 9.13, 10.13, 11.13, 12.13, 13.13, 14.13, 15.13,
        //     0.14, 1.14, 2.14, 3.14, 4.14, 5.14, 6.14, 7.14, 8.14, 9.14, 10.14, 11.14, 12.14, 13.14, 14.14, 15.14,
        //     0.15, 1.15, 2.15, 3.15, 4.15, 5.15, 6.15, 7.15, 8.15, 9.15, 10.15, 11.15, 12.15, 13.15, 14.15, 15.15,
        // ];
        // let b = a;

        let na = ndarray::ArrayView2::from_shape((16, 16), &a).unwrap();
        let nb = ndarray::ArrayView2::from_shape((16, 16), &b).unwrap();

        let mut result_buffer = vec![0.0; 16 * 16];

        let mut kernel = GenericMatrixKernel::<f32, Avx2>::allocate(16);
        unsafe {
            kernel.dot_matrix(
                16,
                16,
                &a,
                16,
                16,
                &b,
                &mut result_buffer,
            );
        }

        let mut expected = ndarray::Array2::zeros((16, 16));
        ndarray::linalg::general_mat_mul(
            1.0,
            &na,
            &nb,
            0.0,
            &mut expected,
        );

        debug_slice("expected", 16, expected.as_slice().unwrap());
        debug_slice("result", 16, &result_buffer);

        assert_eq!(&expected.into_raw_vec(), &result_buffer);
    }

    #[test]
    fn test_basic_f32_gemm_32x32() {
        let (a, b) = generate_sample_data(32 * 32);

        let na = ndarray::ArrayView2::from_shape((32, 32), &a).unwrap();
        let nb = ndarray::ArrayView2::from_shape((32, 32), &b).unwrap();

        let mut result_buffer = vec![0.0; 32 * 32];

        let mut kernel = GenericMatrixKernel::<f32, Avx2>::allocate(32);
        unsafe {
            kernel.dot_matrix(
                32,
                32,
                &a,
                32,
                32,
                &b,
                &mut result_buffer,
            );
        }

        let mut expected = ndarray::Array2::zeros((32, 32));
        ndarray::linalg::general_mat_mul(
            1.0,
            &na,
            &nb,
            0.0,
            &mut expected,
        );

        assert_eq!(&expected.into_raw_vec(), &result_buffer);
    }
}
