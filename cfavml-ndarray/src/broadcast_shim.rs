//! A crate-local copy of ndarray's private `broadcast_with` operation.
//! 
//! Due to the nature of this library, we need to mimic certain behaviours to maintain the
//! 'drop in' nature, which includes mimicking the broadcasting behaviour.
use core::iter::zip;
use ndarray::{ArrayBase, ArrayView, Data, DimMax, ErrorKind, RawData, ShapeError};
use ndarray::Dimension;

#[allow(clippy::type_complexity)]
/// For two arrays or views, find their common shape if possible and
/// broadcast them as array views into that shape.
///
/// Return `ShapeError` if their shapes can not be broadcast together.

pub(crate) fn broadcast_mut_with<'a, 'b, A, B, S, S2, D, E>(
    array_a: &'a ArrayBase<S, D>,
    array_b: &'b ArrayBase<S2, E>,
) -> Result<(ArrayView<'a, A, <D as DimMax<E>>::Output>, ArrayView<'b, B, <D as DimMax<E>>::Output>), ShapeError>
where
    S: RawData<Elem = A> + Data<Elem = A>,
    S2: Data<Elem = B>,
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    let shape = co_broadcast::<D, E, <D as DimMax<E>>::Output>(&array_a.raw_dim(), &array_b.raw_dim())?;
    let view1 = if shape.slice() == array_a.raw_dim().slice() {
        array_a
            .view()
            .into_dimensionality::<<D as DimMax<E>>::Output>()?
    } else if let Some(view1) = array_a.broadcast(shape.clone()) {
        view1
    } else {
        return Err(from_kind(ErrorKind::IncompatibleShape));
    };
    let view2 = if shape.slice() == array_b.raw_dim().slice() {
        array_b
            .view()
            .into_dimensionality::<<D as DimMax<E>>::Output>()?
    } else if let Some(view2) = array_b.broadcast(shape) {
        view2
    } else {
        return Err(from_kind(ErrorKind::IncompatibleShape));
    };
    Ok((view1, view2))
}

/// Calculate the common shape for a pair of array shapes, that they can be broadcasted
/// to. Return an error if the shapes are not compatible.
///
/// Uses the [NumPy broadcasting rules]
//  (https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules).
pub(crate) fn co_broadcast<D1, D2, Output>(shape1: &D1, shape2: &D2) -> Result<Output, ShapeError>
where
    D1: Dimension,
    D2: Dimension,
    Output: Dimension,
{
    let (k, overflow) = shape1.ndim().overflowing_sub(shape2.ndim());
    // Swap the order if d2 is longer.
    if overflow {
        return co_broadcast::<D2, D1, Output>(shape2, shape1);
    }
    // The output should be the same length as shape1.
    let mut out = Output::zeros(shape1.ndim());
    for (out, s) in zip(out.slice_mut(), shape1.slice()) {
        *out = *s;
    }
    for (out, s2) in zip(&mut out.slice_mut()[k..], shape2.slice()) {
        if *out != *s2 {
            if *out == 1 {
                *out = *s2
            } else if *s2 != 1 {
                return Err(from_kind(ErrorKind::IncompatibleShape));
            }
        }
    }
    Ok(out)
}

#[inline]
fn from_kind(error_kind: ErrorKind) -> ShapeError {
    ShapeError::from_kind(error_kind)
}

#[cfg(test)]
mod tests
{
    use ndarray::*;
    use super::*;

    #[test]
    fn test_broadcast_shape()
    {
        fn test_co<D1, D2>(d1: &D1, d2: &D2, r: Result<<D1 as DimMax<D2>>::Output, ShapeError>)
        where
            D1: Dimension + DimMax<D2>,
            D2: Dimension,
        {
            let d = co_broadcast::<D1, D2, <D1 as DimMax<D2>>::Output>(&d1, d2);
            assert_eq!(d, r);
        }
        test_co(&Dim([2, 3]), &Dim([4, 1, 3]), Ok(Dim([4, 2, 3])));
        test_co(&Dim([1, 2, 2]), &Dim([1, 3, 4]), Err(ShapeError::from_kind(ErrorKind::IncompatibleShape)));
        test_co(&Dim([3, 4, 5]), &Ix0(), Ok(Dim([3, 4, 5])));
        let v = vec![1, 2, 3, 4, 5, 6, 7];
        test_co(&Dim(vec![1, 1, 3, 1, 5, 1, 7]), &Dim([2, 1, 4, 1, 6, 1]), Ok(Dim(IxDynImpl::from(v.as_slice()))));
        let d = Dim([1, 2, 1, 3]);
        test_co(&d, &d, Ok(d));
        test_co(&Dim([2, 1, 2]).into_dyn(), &Dim(0), Err(ShapeError::from_kind(ErrorKind::IncompatibleShape)));
        test_co(&Dim([2, 1, 1]), &Dim([0, 0, 1, 3, 4]), Ok(Dim([0, 0, 2, 3, 4])));
        test_co(&Dim([0]), &Dim([0, 0, 0]), Ok(Dim([0, 0, 0])));
        test_co(&Dim(1), &Dim([1, 0, 0]), Ok(Dim([1, 0, 0])));
        test_co(&Dim([1, 3, 0, 1, 1]), &Dim([1, 2, 3, 1]), Err(ShapeError::from_kind(ErrorKind::IncompatibleShape)));
    }
}