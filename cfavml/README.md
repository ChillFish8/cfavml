# CFAVML

> _CF's Accelerated Vector Math Library_

Various accelerated vector operations over Rust primitives with SIMD.

This is the core base library, it has no dependencies and only depends on the `core` library,
it does not perform any allocations.

This library is guaranteed to be no-std compatible and can be adjusted by disabling the `std`
feature flag:

##### Default Setup
```toml
cfavml = "0.2.0" 
```

##### No-std Setup
```toml
cfavml = { version = "0.2.0", default-features = false }
```

### Available SIMD Architectures

- AVX2
- AVX2 + FMA
- AVX512
- NEON
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


### Spacial distances

These are routines that can be used for things like KNN classification or index building.

- Dot product of two vectors
- Cosine distance of two vectors
- Squared Euclidean distance of two vectors

### Arithmetic 

- Add single value to vector
- Sub single value from vector
- Mul vector by single value
- Div vector by single value
- Add two vectors vertically
- Sub two vectors vertically
- Mul two vectors vertically
- Div two vectors vertically

### Comparison

- Horizontal max element in a vector
- Horizontal min element in a vector
- Vertical max element of two vectors
- Vertical min element of two vectors
- Vertical max element of a vector and broadcast value
- Vertical min element of a vector and broadcast value

### Aggregation

- Horizontal sum of a vector

### Misc

- Squared L2 norm of a vector

### Dangerous routine naming convention

If you've looked at the `danger` folder at all, you'll notice a few things, one SIMD operations
are gated behind the `SimdRegister<T>` trait, this provides us with a generic abstraction
over the various SIMD register types and architectures.

This trait, combined with the `Math<T>` trait form the core of all operations and are
provided as generic functions (with no target features):

- `generic_dot`
- `generic_euclidean`
- `generic_cosine`
- `generic_squared_norm`
- `generic_max_horizontal`
- `generic_max_vector`
- `generic_min_horizontal`
- `generic_min_vector`
- `generic_sum`
- `generic_add_value`
- `generic_sub_value`
- `generic_mul_value`
- `generic_div_value`
- `generic_add_vector`
- `generic_sub_vector`
- `generic_mul_vector`
- `generic_div_vector`

We also export functions with the target_features pre-specified for
each SIMD register type and is found under the `cfavml::danger::export_*` 
modules. Although it is not recommended to use these routines directly
unless you know what you are doing.

### Features

- `nightly` Enables optimizations available only on nightly platforms.
  * This is required for AVX512 support due to it currently being unstable.

### Is this a replacement for BLAS?

No. At least, not unless you're only doing dot product... BLAS and LAPACK are _huge_ and I am certainly
not in the market for implementing all BLAS routines in Rust, but that being said if your application is 
similar to that of ndarray where it is only using BLAS for the dot product, then maybe.

