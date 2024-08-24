use core::mem::MaybeUninit;

use ndarray::{Array, ArrayView, DimMax, Dimension};
use cfavml::buffer::WriteOnlyBuffer;
use cfavml::safe_trait_arithmetic_ops::ArithmeticOps;

use crate::utils::{ArrayBackingBuffer, ArrayMutBackingBuffer};

/// Applies the provided buffer operation after broadcasting the array if applicable.
///
/// Unfortunately this route effectively must always return a new owned array and could
/// probably do with being further optimized to minimize allocations.
/// 
/// # Safety
/// 
/// This function takes significant liberties around creating slices to the same mutable object
/// and borrowing because it knows the behaviour of CFAVML and that is what `op` will end up being.
/// 
/// It is _absolutely incorrect_ to use this for ops that do not have identical behaviour to how
/// CFAVML reads and writes memory.
pub(crate) unsafe fn apply_op_with_broadcasting<'a, A, D, E>(
    lhs: Array<A, D>,
    rhs: ArrayView<'a, A, E>,
    op: unsafe fn(&[A], &[A], &mut [MaybeUninit<A>]),
) -> Array<A, <D as DimMax<E>>::Output>
where
    A: ArithmeticOps,
    D: Dimension + DimMax<E>,
    E: Dimension,
    for<'buf> &'buf mut [MaybeUninit<A>]: WriteOnlyBuffer<Item=A>,
{    
    if lhs.ndim() == rhs.ndim() && lhs.shape() == rhs.shape() {
        let mut out = lhs.into_dimensionality().unwrap();

        let mut lhs_buf = crate::utils::get_mut_contiguous_slice_or_reallocate(&mut out);
        let rhs_buf = crate::utils::get_contiguous_slice_or_reallocate(&rhs);

        let lhs_mut_view = lhs_buf.as_mut_slice();
        let rhs_view = rhs_buf.as_slice();

        let lhs_read_view = core::slice::from_raw_parts(lhs_mut_view.as_ptr(), lhs_mut_view.len());
        let lhs_mut_view = core::mem::transmute::<&mut [A], &mut [MaybeUninit<A>]>(lhs_mut_view);
        op(lhs_read_view, rhs_view, lhs_mut_view);

        out
    } else {
        let (lhs_view, rhs_view) = crate::broadcast_shim::broadcast_mut_with(&lhs, &rhs).unwrap();
        if lhs_view.shape() == lhs.shape() {
            let mut out = lhs.into_dimensionality::<<D as DimMax<E>>::Output>().unwrap();

            let mut lhs_buf = crate::utils::get_mut_contiguous_slice_or_reallocate(&mut out);
            let rhs_buf = crate::utils::get_contiguous_slice_or_reallocate(&rhs_view);

            let lhs_mut_view = lhs_buf.as_mut_slice();
            let rhs_view = rhs_buf.as_slice();

            let lhs_read_view = core::slice::from_raw_parts(lhs_mut_view.as_ptr(), lhs_mut_view.len());
            let lhs_mut_view = core::mem::transmute::<&mut [A], &mut [MaybeUninit<A>]>(lhs_mut_view);
            op(lhs_read_view, rhs_view, lhs_mut_view);

            match lhs_buf {
                ArrayMutBackingBuffer::Borrowed(_) => out,
                ArrayMutBackingBuffer::Owned(arr) =>
                    arr.into_dimensionality::<<D as DimMax<E>>::Output>().unwrap(),
            }
        } else if lhs_view.shape() == rhs_view.shape() {
            let target_shape = lhs_view.raw_dim();
            
            let lhs_buf = crate::utils::get_contiguous_slice_or_reallocate(&lhs_view);
            let rhs_buf = crate::utils::get_contiguous_slice_or_reallocate(&rhs_view);

            // An attempt to intelligently handle buffers and try to avoid needlessly re-allocating.
            match (lhs_buf, rhs_buf) {
                // The middle ground, we need to do _an_ allocation so we might as well allocate an
                // uninitialized buffer and write directly to that.
                (ArrayBackingBuffer::Borrowed(lhs_view), ArrayBackingBuffer::Borrowed(rhs_view)) => {
                    let mut output = Vec::<MaybeUninit<A>>::with_capacity(lhs_view.len());
                    output.set_len(lhs_view.len());
                    
                    op(lhs_view, rhs_view, &mut output);
                        
                    let initialized = core::mem::transmute::<Vec<MaybeUninit<A>>, Vec<A>>(output);
                    Array::from_shape_vec_unchecked(target_shape, initialized) 
                },
                // The LHS buffer is owned, so we can just mutate the buffer within.
                (ArrayBackingBuffer::Owned(mut lhs_arr), rhs_buf) => {
                    let lhs_mut_view = lhs_arr.as_slice_memory_order_mut().unwrap();
                    let rhs_view = rhs_buf.as_slice();

                    let lhs_read_view = core::slice::from_raw_parts(lhs_mut_view.as_ptr(), lhs_mut_view.len());
                    let lhs_mut_view = core::mem::transmute::<&mut [A], &mut [MaybeUninit<A>]>(lhs_mut_view);
                    op(lhs_read_view, rhs_view, lhs_mut_view);
                                        
                    lhs_arr
                },
                // The RHS buffer is owned, so we can just mutate the buffer within.
                (lhs_buf, ArrayBackingBuffer::Owned(mut rhs_arr)) => {
                    let lhs_view = lhs_buf.as_slice();
                    let rhs_mut_view = rhs_arr.as_slice_memory_order_mut().unwrap();

                    let rhs_read_view = core::slice::from_raw_parts(rhs_mut_view.as_ptr(), rhs_mut_view.len());
                    let rhs_mut_view = core::mem::transmute::<&mut [A], &mut [MaybeUninit<A>]>(rhs_mut_view);
                    op(lhs_view, rhs_read_view, rhs_mut_view);

                    rhs_arr
                },                
            }
        } else {
            // TODO: Can this even occur?
            unimplemented!(
                "Congratulations! You found and edge case I didn't think of, please report \
                    this to the CFAVML repository on GitHub, \
                    CFAVML currently cannot correctly handle mismatching broadcast sizes"
            );
        }
    }
}

/// Applies the provided op across each element in the array with the given broadcast value.
///
/// # Safety
///
/// This function takes significant liberties around creating slices to the same mutable object
/// and borrowing because it knows the behaviour of CFAVML and that is what `op` will end up being.
///
/// It is _absolutely incorrect_ to use this for ops that do not have identical behaviour to how
/// CFAVML reads and writes memory.
pub(crate) unsafe fn apply_op_with_broadcast_value<'a, A, D>(
    mut lhs: Array<A, D>,
    value: A,
    op: unsafe fn(A, &[A], &mut [MaybeUninit<A>]),
) -> Array<A, D>
where
    A: ArithmeticOps,
    D: Dimension,
    for<'buf> &'buf mut [MaybeUninit<A>]: WriteOnlyBuffer<Item=A>,
{
    let mut lhs_buf = crate::utils::get_mut_contiguous_slice_or_reallocate(&mut lhs);
    let lhs_mut_view = lhs_buf.as_mut_slice();

    let lhs_read_view = core::slice::from_raw_parts(lhs_mut_view.as_ptr(), lhs_mut_view.len());
    let lhs_mut_view = core::mem::transmute::<&mut [A], &mut [MaybeUninit<A>]>(lhs_mut_view);
    
    op(value, lhs_read_view, lhs_mut_view);
    
    lhs
}

    #[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2, Array3, Axis, Slice};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use super::*;
    
    #[test]
    fn test_same_shape_op() {
        let a = Array3::random((3, 3, 3), Uniform::new(0.0, 10.0));
        let b = Array3::random((3, 3, 3), Uniform::new(0.0, 10.0));

        let expected = &a + &b;                        
        let res = unsafe { apply_op_with_broadcasting(a, b.view(), cfavml::add_vector) };
        assert_eq!(expected, res);
    }
    
    #[test]
    fn test_broadcast_to_lhs_shape() {
        let a = Array3::random((3, 3, 3), Uniform::new(0.0, 10.0));
        let b = Array3::random((1, 3, 3), Uniform::new(0.0, 10.0));

        let expected = &a + &b;
        let res = unsafe { apply_op_with_broadcasting(a, b.view(), cfavml::add_vector) };
        assert_eq!(res.shape(), expected.shape());
        assert_eq!(expected, res);        
    }
    
    #[test]
    fn test_broadcast_to_new_shape() {
        let a = Array2::random((4, 1), Uniform::new(0.0, 10.0));
        let b = Array1::random((3,), Uniform::new(0.0, 10.0));

        let expected = &a + &b;
        let res = unsafe { apply_op_with_broadcasting(a, b.view(), cfavml::add_vector) };
        assert_eq!(res.shape(), expected.shape());
        assert_eq!(expected, res);
    }

    #[test]
    fn test_broadcast_from_arr_slices() {
        let a = Array2::random((4, 4), Uniform::new(0.0, 10.0));
        let b = Array2::random((3, 3), Uniform::new(0.0, 10.0));

        let slice_a = a.slice_axis(Axis(0), Slice::new(0, None, 2));
        let slice_b_tmp = b.slice_axis(Axis(1), Slice::new(0, None, 4));
        let slice_b = slice_b_tmp.slice_axis(Axis(0), Slice::new(0, None, 2));

        let expected = &slice_a + &slice_b;
        let res = unsafe { apply_op_with_broadcasting(slice_a.to_owned(), slice_b.view(), cfavml::add_vector) };
        assert_eq!(res.shape(), expected.shape());
        assert_eq!(expected, res);
    }
}