use crate::buffer::WriteOnlyBuffer;
use crate::danger::{DenseLane, SimdRegister};
use crate::math::Math;
use crate::mem_loader::{IntoMemLoader, MemLoader};

#[inline(always)]
pub(crate) unsafe fn apply_vertical_kernel<T, R, M, B1, B2, B3>(
    a: B1,
    b: B2,
    mut result: &mut [B3],
    dense_lane_kernel: unsafe fn(
        DenseLane<R::Register>,
        DenseLane<R::Register>,
    ) -> DenseLane<R::Register>,
    reg_kernel: unsafe fn(R::Register, R::Register) -> R::Register,
    single_kernel: unsafe fn(T, T) -> T,
) where
    T: Copy,
    R: SimdRegister<T>,
    M: Math<T>,
    B1: IntoMemLoader<T>,
    B1::Loader: MemLoader<Value = T>,
    B2: IntoMemLoader<T>,
    B2::Loader: MemLoader<Value = T>,
    for<'a> &'a mut [B3]: WriteOnlyBuffer<Item = T>,
{
    let project_to_len = result.raw_buffer_len();
    let result_ptr = result.as_write_only_ptr();

    let mut a = a.into_projected_mem_loader(project_to_len);
    let mut b = b.into_projected_mem_loader(project_to_len);

    let offset_from = project_to_len % R::elements_per_dense();

    // Operate over dense lanes first.
    let mut i = 0;
    while i < (project_to_len - offset_from) {
        let l1 = a.load_dense::<R>();
        let l2 = b.load_dense::<R>();
        let max = dense_lane_kernel(l1, l2);
        R::write_dense(result_ptr.add(i), max);

        i += R::elements_per_dense();
    }

    // Operate over single registers next.
    let offset_from = offset_from % R::elements_per_lane();
    while i < (project_to_len - offset_from) {
        let l1 = a.load::<R>();
        let l2 = b.load::<R>();
        let max = reg_kernel(l1, l2);
        R::write(result_ptr.add(i), max);

        i += R::elements_per_lane();
    }

    while i < project_to_len {
        result.write_at(i, single_kernel(a.read(), b.read()));

        i += 1;
    }
}
