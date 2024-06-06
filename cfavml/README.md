# eonn-accel

Various specialised vector operations, primarily designed for similarity search.

This system has several specializations for various CPU features and vector dimensions,
currently only `f32` vectors with dimensions of  any size are supported, but with 
specialized const generic variants for various multiples of 128 or 64 (depending on arch).

Supported CPU features include `Avx512`, `Avx2` and `Fma`, fallback implementations can
be optimized relatively well by the compiler for other architectures e.g. ARM or SSE.

### Supported Operations & Distances

- `dot(a, b)`
- `norm(a)`  - Equivalent to `np.inner()` (squared norm)
- `cosine(a, b)`     - Does not do inverse by itself
- `euclidean(a, b)`  - Squared euclidean
- `div(a, value)` - Vector x single-value
- `mul(a, value)` - Vector x single-value
- `add(a, value)` - Vector x single-value
- `sub(a, value)` - Vector x single-value
- `div(a, b)` - Vector x vector
- `mul(a, b)` - Vector x vector
- `add(a, b)` - Vector x vector
- `sub(a, b)` - Vector x vector
- `sum_horizontal(a)`
- `max_horizontal(a)`
- `min_horizontal(a)`
- `sum_vertical(m)` - 2D matrix
- `max_vertical(m)` - 2D matrix
- `min_vertical(m)` - 2D matrix

### Dangerous routine naming convention

If you've looked at the `danger` folder at all, you'll notice all functions implement a certain
naming scheme to indicate their specialization.

```
<dtype>_x<dims>_<arch>_<(no)fma>_<op_name>
```

### Features

- `nightly` Enables optimizations available only on nightly platforms.
  * Fallback implementations may see much better performance.
  * This is required for AVX512 support due to it currently being unstable.

