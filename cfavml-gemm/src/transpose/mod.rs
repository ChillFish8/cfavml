use std::any::TypeId;
use std::mem;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod impl_avx2_32bit;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod impl_avx2_64bit;

/// Transpose a given matrix, writing the result to the given output buffer.
pub fn transpose_matrix<T>(width: usize, height: usize, data: &[T], result: &mut [T])
where
    T: Copy + 'static,
{
    assert_eq!(data.len(), width * height, "Input data shape missmatch");
    assert_eq!(
        data.len(),
        result.len(),
        "Output buffer does not match input data"
    );

    if width == 0 || height == 0 {
        return;
    }

    if width == 1 || height == 1 {
        result.copy_from_slice(data);
        return;
    }

    if TypeId::of::<T>() == TypeId::of::<f32>()
        || TypeId::of::<T>() == TypeId::of::<u32>()
    {
        let data = unsafe { mem::transmute::<&[T], &[f32]>(data) };
        let result = unsafe { mem::transmute::<&mut [T], &mut [f32]>(result) };

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if is_x86_feature_detected!("avx2") {
                impl_avx2_32bit::transpose_32bit(width, height, data, result)
            }
        }
    } else if TypeId::of::<T>() == TypeId::of::<f64>()
        || TypeId::of::<T>() == TypeId::of::<u64>()
    {
        let data = unsafe { mem::transmute::<&[T], &[f64]>(data) };
        let result = unsafe { mem::transmute::<&mut [T], &mut [f64]>(result) };

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if is_x86_feature_detected!("avx2") {
                impl_avx2_64bit::transpose_64bit(width, height, data, result)
            }
        }
    }
}
