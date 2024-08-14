# CFAVML GEMM

> _BLAS-like general matrix multiplication extension for `cfavml`_

This provides additional routines for general matmul operations via the `cfavml` library.

The reason for this being a separate crate is that I wanted to detach the core library
from any dependencies or std lib dependency. 

This library generally tries to be as optimized as possible for the same architectures
as `cfavml`, it may sometimes be equivalent performance as OpenBLAS, other times it may be 
slower.

## Available Methods

##### Generic impls

- `generic_matrix_multiply` - `Matrix @ Matrix `
- `generic_matrix_vector_multiply` - `Matrix @ Vector`

##### Exported non-generic impls

**Dynamic Size**
- `f32_xany_avx512_gemm`
- `f64_xany_avx512_gemm`
- `f32_xany_avx2_gemm`
- `f64_xany_avx2_gemm`
- `f32_xany_avx2fma_gemm`
- `f64_xany_avx2fma_gemm`
- `f32_xany_neon_gemm`
- `f64_xany_neon_gemm`
- `f32_xany_fallback_gemm`
- `f64_xany_fallback_gemm`

**Const Size**
- `f32_xconst_avx512_gemm`
- `f64_xconst_avx512_gemm`
- `f32_xconst_avx2_gemm`
- `f64_xconst_avx2_gemm`
- `f32_xconst_avx2fma_gemm`
- `f64_xconst_avx2fma_gemm`
- `f32_xconst_neon_gemm`
- `f64_xconst_neon_gemm`
- `f32_xconst_fallback_gemm`
- `f64_xconst_fallback_gemm`

## Notes on Threading

This library has multi-threading inbuilt into it, all threads a pinned to specific cores
in order to avoid CPU cache issues but this may impact other programs running at the same time.

This system will implicitly look for the following env vars to configure the CPU usage limit:

(In priority order)

- `CFAVML_NUM_THREADS`
- `OMP_NUM_THREADS`  (Enable feature `env-var-compat`)
- `OPENBLAS_NUM_THREADS`  (Enable feature `env-var-compat`)

If no env var is provided, the system will use the number of **Physical** CPU cores available
as provided by the `num_cpus` crate.
