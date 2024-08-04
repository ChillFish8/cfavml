# CFAVML

> _CF's Accelerated Vector Math Library_

Various accelerated vector operations over Rust primitives with SIMD.

### Available SIMD Architectures

- AVX2
- AVX2 + FMA
- AVX512
- NEON
- Fallback (Typically optimized to SSE automatically by LLVM on x86)

## Crates

### `cfavml` 

This is the core base library, it has no dependencies and only depends on the `core` library,
it does not perform any allocations.

This library is guaranteed to be no-std compatible and can be adjusted by disabling the `std`
feature flag:

##### Default Setup
```toml
cfavml = "0.1.0" 
```

##### No-std Setup
```toml
cfavml = { version = "0.1.0", default-features = false }
```

### `cfavml-gemm`

Generic matrix multiplication routines + transposition.

This crate is a WIP and is not currently published.

You should not use this crate currently.

### `cfavml-utils`

This crate is a WIP and is not currently published.

Are a set of utilities primarily designed to just be use by the other cfavml sub-crates. 
This contains a lightly wrapped rayon threadpool with CPU pinning and provides the ability 
to create aligned buffers.

NOTE:

This library is _not_ no-std compatible and _does_ allocate.

### Is this a replacement for BLAS?

No. At least, not unless you're only doing dot product... BLAS and LAPACK are _huge_ and I am certainly
not in the market for implementing all BLAS routines in Rust, but that being said if your application is
similar to that of ndarray where it is only using BLAS for the dot product, then maybe.
