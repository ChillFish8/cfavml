# eonn-accel

Various specialised vector operations, primarily designed for similarity search.

This system has several specializations for various CPU features and vector dimensions,
currently only `f32` vectors with dimensions of  any size are supported, but with 
specialized `512`, `768` or `1024` dimension implementations.

Supported CPU features include `Avx512`, `Avx2` and `Fma`, fallback implementations can
be optimized relatively well by the compiler for other architectures e.g. ARM or SSE.

### Supported Operations & Distances

- `vector.dot(other)`
- `vector.dist_dot(other)`
- `vector.dist_cosine(other)`
- `vector.dist_squared_euclidean(other)`
- `vector.angular_hyperplane(other)`
- `vector.euclidean_hyperplane(other)`
- `vector.squared_norm()`
- `vector / value`
- `vector * value`
- `vector + value`
- `vector - value`
- `vector / vector`
- `vector * vector`
- `vector + vector`
- `vector - vector`
- `vector.sum()`
- `vector.max()`
- `vector.min()`
- `vector.mean()`
- `[vector].vertical_min()`   ~ Unsafe API only currently (Sorry)
- `[vector].vertical_max()`   ~ Unsafe API only currently (Sorry)
- `[vector].vertical_sum()`   ~ Unsafe API only currently (Sorry)
- `[vector].vertical_mean()`  ~ Unsafe API only currently (Sorry)

### Dangerous routine naming convention

If you've looked at the `danger` folder at all, you'll notice all functions implement a certain
naming scheme to indicate their specialization.

```
<dtype>_x<dims>_<arch>_<(no)fma>_<op_name>
```

### Features

- `dangerous-access` Exposes access to the unsafe specialized functions, it is entirely on you to 
  ensure the data passed to these functions are correct and safe to call. USE AT YOUR OWN RISK.
- `nightly` Enables optimizations available only on nightly platforms.
  * Fallback implementations may see much better performance.
  * This is required for AVX512 support due to it currently being unstable.

