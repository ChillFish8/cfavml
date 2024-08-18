use crate::buffer::WriteOnlyBuffer;
use crate::danger::{DenseLane, SimdRegister};


#[inline(always)]
pub(crate) unsafe fn apply_vector_x_value_kernel<T, R, B>(
    dims: usize,
    value: T,
    a: &[T],
    mut result: &mut [B],
    dense_lane_kernel: unsafe fn(
        DenseLane<R::Register>,
        DenseLane<R::Register>,
    ) -> DenseLane<R::Register>,
    reg_kernel: unsafe fn(R::Register, R::Register) -> R::Register,
    single_kernel: unsafe fn(T, T) -> T,
) where
    T: Copy,
    R: SimdRegister<T>,
    for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
{
    debug_assert_eq!(a.len(), dims, "Vector `a` does not match size `dims`");
    debug_assert_eq!(
        result.raw_buffer_len(),
        dims,
        "Vector `result` does not match size `dims`"
    );

    let value_reg = R::filled(value);
    let value_dense = DenseLane::copy(value_reg);

    let offset_from = dims % R::elements_per_dense();
    let a_ptr = a.as_ptr();
    let result_ptr = result.as_write_only_ptr();

    // Operate over dense lanes first.
    let mut i = 0;
    while i < (dims - offset_from) {
        let l1 = R::load_dense(a_ptr.add(i));
        let mask = dense_lane_kernel(l1, value_dense);
        R::write_dense(result_ptr.add(i), mask);

        i += R::elements_per_dense();
    }

    // Operate over single registers next.
    let offset_from = offset_from % R::elements_per_lane();
    while i < (dims - offset_from) {
        let l1 = R::load(a_ptr.add(i));
        let mask = reg_kernel(l1, value_reg);
        R::write(result_ptr.add(i), mask);

        i += R::elements_per_lane();
    }

    while i < dims {
        let a = *a.get_unchecked(i);
        result.write_at(i, single_kernel(a, value));

        i += 1;
    }
}

#[inline(always)]
pub(crate) unsafe fn apply_vector_x_vector_kernel<T, R, B>(
    dims: usize,
    a: &[T],
    b: &[T],
    mut result: &mut [B],
    dense_lane_kernel: unsafe fn(
        DenseLane<R::Register>,
        DenseLane<R::Register>,
    ) -> DenseLane<R::Register>,
    reg_kernel: unsafe fn(R::Register, R::Register) -> R::Register,
    single_kernel: unsafe fn(T, T) -> T,
) where
    T: Copy,
    R: SimdRegister<T>,
    for<'a> &'a mut [B]: WriteOnlyBuffer<Item = T>,
{
    debug_assert_eq!(a.len(), dims, "Vector `a` does not match size `dims`");
    debug_assert_eq!(b.len(), dims, "Vector `b` does not match size `dims`");
    debug_assert_eq!(
        result.raw_buffer_len(),
        dims,
        "Vector `result` does not match size `dims`"
    );

    let offset_from = dims % R::elements_per_dense();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let result_ptr = result.as_write_only_ptr();

    // Operate over dense lanes first.
    let mut i = 0;
    while i < (dims - offset_from) {
        let l1 = R::load_dense(a_ptr.add(i));
        let l2 = R::load_dense(b_ptr.add(i));
        let res = dense_lane_kernel(l1, l2);
        R::write_dense(result_ptr.add(i), res);

        i += R::elements_per_dense();
    }

    // Operate over single registers next.
    let offset_from = offset_from % R::elements_per_lane();
    while i < (dims - offset_from) {
        let l1 = R::load(a_ptr.add(i));
        let l2 = R::load(b_ptr.add(i));
        let res = reg_kernel(l1, l2);
        R::write(result_ptr.add(i), res);

        i += R::elements_per_lane();
    }

    while i < dims {
        let a = *a.get_unchecked(i);
        let b = *b.get_unchecked(i);
        result.write_at(i, single_kernel(a, b));

        i += 1;
    }
}
