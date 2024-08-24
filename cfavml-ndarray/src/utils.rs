use core::mem::ManuallyDrop;
use ndarray::{Array, ArrayBase, Data, DataMut, Dimension};
use cfavml::safe_trait_arithmetic_ops::ArithmeticOps;

/// Transmute from A to B.
///
/// Like transmute, but does not have the compile-time size check which blocks
/// using regular transmute in some cases.
///
/// **Panics** if the size of A and B are different.
#[track_caller]
#[inline]
pub(crate) unsafe fn unlimited_transmute<A, B>(data: A) -> B
{
    // safe when sizes are equal and caller guarantees that representations are equal
    assert_eq!(size_of::<A>(), size_of::<B>());
    let old_data = ManuallyDrop::new(data);
    (&*old_data as *const A as *const B).read()
}

#[inline]
/// Attempts to extract the underlying slice of memory backing the array if it is contiguous,
/// otherwise allocate the elements in a contiguous buffer and return the value.
pub(crate) fn get_mut_contiguous_slice_or_reallocate<S, A, D>(
    arr: &mut ArrayBase<S, D>
) -> ArrayMutBackingBuffer<A, D> 
where
    A: Clone + ArithmeticOps,
    S: Data<Elem=A> + DataMut,
    D: Dimension,
{
    let fake_reference_without_borrow_limitation = arr as *const ArrayBase<S, D>;
    
    if let Some(buf) = arr.as_slice_memory_order_mut() {
        return ArrayMutBackingBuffer::Borrowed(buf);
    }
    
    let owned = unsafe { 
        let temporary_new_reference = &*fake_reference_without_borrow_limitation;
        temporary_new_reference.to_owned() 
    };

    ArrayMutBackingBuffer::Owned(owned)
}

#[inline]
/// Attempts to extract the underlying slice of memory backing the array if it is contiguous,
/// otherwise allocate the elements in a contiguous buffer and return the value.
pub(crate) fn get_contiguous_slice_or_reallocate<S, A, D>(
    arr: &ArrayBase<S, D>
) -> ArrayBackingBuffer<A, D>
where
    A: Clone + ArithmeticOps,
    S: Data<Elem=A>,
    D: Dimension,
{
    let fake_reference_without_borrow_limitation = arr as *const ArrayBase<S, D>;

    if let Some(buf) = arr.as_slice_memory_order() {
        return ArrayBackingBuffer::Borrowed(buf);
    }

    let owned = unsafe {
        let temporary_new_reference = &*fake_reference_without_borrow_limitation;
        temporary_new_reference.to_owned()
    };

    ArrayBackingBuffer::Owned(owned)
}

pub(crate) enum ArrayMutBackingBuffer<'a, A, D> {
    Borrowed(&'a mut [A]),
    Owned(Array<A, D>),
}

impl<'a, A, D> ArrayMutBackingBuffer<'a, A, D>
where
    D: Dimension,
{
    pub(crate) fn as_mut_slice(&mut self) -> &mut [A] {
        match self {
            Self::Borrowed(buf) => buf,
            Self::Owned(buf) => buf.as_slice_memory_order_mut()
                .expect("Buffer should be contiguous"),
        }
    }
}

pub(crate) enum ArrayBackingBuffer<'a, A, D> {
    Borrowed(&'a [A]),
    Owned(Array<A, D>),
}

impl<'a, A, D> ArrayBackingBuffer<'a, A, D>
where
    D: Dimension,
{
    pub(crate) fn as_slice(&self) -> &[A] {
        match self {
            Self::Borrowed(buf) => buf,
            Self::Owned(buf) => buf.as_slice_memory_order()
                .expect("Buffer should be contiguous"),
        }
    }
}


#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;
    use super::*;
    
    #[test]
    fn test_mut_owned_array_contiguous() {
        let mut arr = Array2::random((4, 4), Uniform::new(0., 10.));
        let buffer = get_mut_contiguous_slice_or_reallocate(&mut arr);
        assert!(matches!(buffer, ArrayMutBackingBuffer::Borrowed(_)), "Buffer should be using borrowed");
    }

    #[test]
    fn test_mut_view_array_contiguous() {
        let mut arr = Array2::random((4, 4), Uniform::new(0., 10.));
        let mut view = arr.view_mut();
        let buffer = get_mut_contiguous_slice_or_reallocate(&mut view);
        assert!(matches!(buffer, ArrayMutBackingBuffer::Borrowed(_)), "Buffer should be using borrowed");
    }
    
    #[test]
    fn test_mut_arc_array_contiguous() {  // Arc is COW.
        let mut arr = Array2::random((4, 4), Uniform::new(0., 10.)).into_shared();
        let buffer = get_mut_contiguous_slice_or_reallocate(&mut arr);
        assert!(matches!(buffer, ArrayMutBackingBuffer::Borrowed(_)), "Buffer should be owned");
    }

    #[test]
    fn test_owned_array_contiguous() {
        let arr = Array2::random((4, 4), Uniform::new(0., 10.));
        let buffer = get_contiguous_slice_or_reallocate(&arr);
        assert!(matches!(buffer, ArrayBackingBuffer::Borrowed(_)), "Buffer should be using borrowed");
    }

    #[test]
    fn test_view_array_contiguous() {
        let arr = Array2::random((4, 4), Uniform::new(0., 10.));
        let basic_view = arr.view();
        let buffer = get_contiguous_slice_or_reallocate(&basic_view);
        assert!(matches!(buffer, ArrayBackingBuffer::Borrowed(_)), "Buffer should be using borrowed");
    }

    #[test]
    fn test_broadcast_array_non_contiguous() {
        let arr = Array1::random((4,), Uniform::new(0., 10.));
        let basic_view = arr
            .broadcast((4, 4))
            .expect("Broadcast shape");
        let buffer = get_contiguous_slice_or_reallocate(&basic_view);
        assert!(matches!(buffer, ArrayBackingBuffer::Owned(_)), "Buffer should be using borrowed");
    }
    
    #[test]
    fn test_arc_array_contiguous() {  // Arc is COW.
        let arr = Array2::random((4, 4), Uniform::new(0., 10.)).into_shared();
        let buffer = get_contiguous_slice_or_reallocate(&arr);
        assert!(matches!(buffer, ArrayBackingBuffer::Borrowed(_)), "Buffer should be owned");
    }
}