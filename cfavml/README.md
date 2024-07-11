# CFAVML

> _CF's Accelerated Vector Math Library_

Various accelerated vector operations over Rust primitives with SIMD.

### Available SIMD Architectures

- AVX2
- AVX2 + FMA
- AVX512
- NEON (`f32`/`f64` operations only currently)
- Fallback (Typically optimized to SSE automatically by LLVM on x86)

### Supported Primitives

- `f32`
- `f64`
- `i8`
- `i16`
- `i32`
- `i64`
- `u8`
- `u16`
- `u32`
- `u64`

##### Note on non-`f32/f64` division

Division operations on non-floating point primitives are currently still scalar
operations, as performing integer division is incredibly hard to do anymore efficiently
with SIMD and adds a significant amount of cognitive overhead when reading the code.

Although to be honest I have some serious questions about your application if you're doing 
heavy integer division...

### Supported Operations & Distances

- Dot Product
- L2 Norm
- Squared Euclidean
- Cosine 
- Vector Min Value
- Vector Max Value
- Vector Sum Value
- Vector Add Value
- Vector Sub Value
- Vector Mul Value
- Vector Div Value
- Vector Add Vector
- Vector Sub Vector
- Vector Mul Vector
- Vector Div Vector

### Dangerous routine naming convention

If you've looked at the `danger` folder at all, you'll notice a few things, one SIMD operations
are gated behind the `SimdRegister<T>` trait, this provides us with a generic abstraction
over the various SIMD register types and architectures.

This trait, combined with the `Math<T>` trait form the core of all operations and are
provided as generic functions (with no target features):

- `generic_dot_product`
- `generic_euclidean`
- `generic_cosine`
- `generic_norm`
- `generic_max_horizontal`
- `generic_max_vertical`
- `generic_min_horizontal`
- `generic_min_vertical`
- `generic_sum_horizontal`
- `generic_sum_vertical`
- `generic_add_value`
- `generic_sub_value`
- `generic_mul_value`
- `generic_div_value`
- `generic_add_vector`
- `generic_sub_vector`
- `generic_mul_vector`
- `generic_div_vector`
- `generic_matmul_matrix` - This single handily is the cause of much of my pain. 

We also provide pre-configured non-generic methods with have the relevant `target_feature`s 
specified, naturally these methods are immediately UB if you call them without the correct
CPU flags being available.

```no_test
<dtype>_x<dims>_<arch>_<(no)fma>_<op_name>
```

### Features

- `nightly` Enables optimizations available only on nightly platforms.
  * This is required for AVX512 support due to it currently being unstable.

### Is this a replacement for BLAS?

No. At least, not unless you're only doing dot product... BLAS and LAPACK are _huge_ and I am certainly
not in the market for implementing all BLAS routines in Rust, but that being said if your application is 
similar to that of ndarray where it is only using BLAS for the dot product, then maybe.

