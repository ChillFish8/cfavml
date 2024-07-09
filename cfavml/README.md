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
  * This is required for AVX512 support due to it currently being unstable.

### Benchmarks - Intel

Ran on a Hetzner `CX52 Intel x86`.

CPU Supports `AVX512`, `AVX2` and the `SSE` families.

Ndarray compiled with openblas installed via `libopenblas-dev`. 
`OMP_NUM_THRESD=1`

```
ndarray x1024 dot                       time:   [68.526 µs 69.200 µs 69.933 µs]
ndarray x1024 cosine                    time:   [185.08 µs 186.51 µs 187.95 µs]
ndarray x1024 euclidean                 time:   [433.85 µs 437.34 µs 440.88 µs]
                                        
simsimd x1024 dot                       time:   [91.877 µs 92.306 µs 92.764 µs]
simsimd x1024 cosine                    time:   [121.50 µs 122.78 µs 124.57 µs]
simsimd x1024 euclidean                 time:   [102.72 µs 103.94 µs 105.16 µs]
                                        
f32_avx2_nofma_dot x1024                time:   [87.033 µs 87.549 µs 88.096 µs]
f32_avx2_nofma_dot xany-1301            time:   [107.18 µs 107.73 µs 108.33 µs]
f32_avx2_nofma_dot xany-1024            time:   [80.020 µs 80.397 µs 80.732 µs]
                                        
f32_avx2_fma_dot x1024                  time:   [79.109 µs 79.440 µs 79.811 µs]
f32_avx2_fma_dot xany-1301              time:   [86.501 µs 87.004 µs 87.507 µs]
f32_avx2_fma_dot xany-1024              time:   [81.079 µs 81.399 µs 81.734 µs]
                                        
f32_avx2_nofma_cosine x1024             time:   [275.73 µs 277.49 µs 279.39 µs]
f32_avx2_nofma_cosine xany-1301         time:   [414.16 µs 422.72 µs 433.58 µs]
f32_avx2_nofma_cosine xany-1024         time:   [341.22 µs 343.26 µs 345.30 µs]
                                        
f32_avx2_fma_cosine x1024               time:   [216.62 µs 218.46 µs 220.55 µs]
f32_avx2_fma_cosine xany-1301           time:   [343.44 µs 352.89 µs 362.38 µs]
f32_avx2_fma_cosine xany-1024           time:   [295.30 µs 300.96 µs 307.62 µs]

f32_avx2_nofma_euclidean x1024          time:   [122.87 µs 125.17 µs 127.50 µs]
f32_avx2_nofma_euclidean xany-1301      time:   [147.26 µs 149.43 µs 151.98 µs]
f32_avx2_nofma_euclidean xany-1024      time:   [115.04 µs 116.49 µs 118.05 µs]
                                        
f32_avx2_fma_euclidean x1024            time:   [85.604 µs 86.501 µs 87.524 µs]
f32_avx2_fma_euclidean xany-1301        time:   [134.52 µs 136.73 µs 139.56 µs]
f32_avx2_fma_euclidean xany-1024        time:   [96.740 µs 97.632 µs 98.708 µs]

f32_fallback_nofma_dot x1024            time:   [452.12 µs 454.53 µs 456.81 µs]
f32_fallback_nofma_dot xany-1301        time:   [578.87 µs 583.11 µs 587.90 µs]
f32_fallback_nofma_dot xany-1024        time:   [443.31 µs 448.21 µs 452.48 µs]

f32_fallback_nofma_cosine x1024         time:   [1.1164 ms 1.1224 ms 1.1287 ms]
f32_fallback_nofma_cosine xany-1301     time:   [1.4166 ms 1.4220 ms 1.4274 ms]
f32_fallback_nofma_cosine xany-1024     time:   [1.1336 ms 1.1434 ms 1.1545 ms]

f32_fallback_nofma_euclidean x1024      time:   [526.37 µs 537.05 µs 548.41 µs]
f32_fallback_nofma_euclidean xany-1301  time:   [662.82 µs 672.44 µs 682.99 µs]
f32_fallback_nofma_euclidean xany-1024  time:   [511.62 µs 521.75 µs 532.44 µs]
```

### Benchmarks - AMD

Ran on a Hetzner `CPX51 AMD x86`.

CPU Supports `AVX2` and the `SSE` families.

Ndarray compiled with openblas installed via `libopenblas-dev`.
`OMP_NUM_THRESD=1`

```
ndarray x1024 dot                       time:   [60.558 µs 60.929 µs 61.283 µs]
ndarray x1024 cosine                    time:   [184.70 µs 185.91 µs 187.15 µs]
ndarray x1024 euclidean                 time:   [238.59 µs 240.13 µs 241.69 µs]

simsimd x1024 dot                       time:   [897.33 µs 902.23 µs 906.68 µs]
simsimd x1024 cosine                    time:   [933.19 µs 939.19 µs 944.96 µs]
simsimd x1024 euclidean                 time:   [896.06 µs 901.35 µs 906.54 µs]

f32_avx2_nofma_dot x1024                time:   [57.765 µs 58.056 µs 58.328 µs]
f32_avx2_nofma_dot xany-1301            time:   [77.447 µs 77.948 µs 78.440 µs]
f32_avx2_nofma_dot xany-1024            time:   [56.186 µs 56.580 µs 56.944 µs]

f32_avx2_fma_dot x1024                  time:   [40.432 µs 40.759 µs 41.096 µs]
f32_avx2_fma_dot xany-1301              time:   [71.430 µs 72.013 µs 72.575 µs]
f32_avx2_fma_dot xany-1024              time:   [41.927 µs 42.206 µs 42.482 µs]

f32_avx2_nofma_cosine x1024             time:   [185.24 µs 186.55 µs 187.87 µs]
f32_avx2_nofma_cosine xany-1301         time:   [241.52 µs 243.23 µs 244.98 µs]
f32_avx2_nofma_cosine xany-1024         time:   [193.88 µs 195.33 µs 196.81 µs]

f32_avx2_fma_cosine x1024               time:   [194.12 µs 195.54 µs 196.95 µs]
f32_avx2_fma_cosine xany-1301           time:   [259.67 µs 261.11 µs 262.47 µs]
f32_avx2_fma_cosine xany-1024           time:   [203.31 µs 204.75 µs 206.12 µs]

f32_avx2_nofma_euclidean x1024          time:   [51.604 µs 51.913 µs 52.216 µs]
f32_avx2_nofma_euclidean xany-1301      time:   [77.431 µs 77.979 µs 78.531 µs]
f32_avx2_nofma_euclidean xany-1024      time:   [56.036 µs 56.435 µs 56.813 µs]

f32_avx2_fma_euclidean x1024            time:   [63.379 µs 63.756 µs 64.109 µs]
f32_avx2_fma_euclidean xany-1301        time:   [72.760 µs 73.246 µs 73.707 µs]
f32_avx2_fma_euclidean xany-1024        time:   [52.056 µs 52.448 µs 52.847 µs]

f32_fallback_nofma_dot x1024            time:   [319.21 µs 321.19 µs 323.09 µs]
f32_fallback_nofma_dot xany-1301        time:   [401.54 µs 404.57 µs 407.62 µs]
f32_fallback_nofma_dot xany-1024        time:   [320.53 µs 322.66 µs 324.57 µs]

f32_fallback_nofma_cosine x1024         time:   [715.65 µs 719.69 µs 723.60 µs]
f32_fallback_nofma_cosine xany-1301     time:   [894.42 µs 899.93 µs 905.42 µs]
f32_fallback_nofma_cosine xany-1024     time:   [702.34 µs 706.97 µs 711.35 µs]

f32_fallback_nofma_euclidean x1024      time:   [346.84 µs 348.83 µs 350.79 µs]
f32_fallback_nofma_euclidean xany-1301  time:   [432.80 µs 435.61 µs 438.26 µs]
f32_fallback_nofma_euclidean xany-1024  time:   [339.50 µs 341.79 µs 344.09 µs]
```

### Benchmarks - ARM

Ran on a Hetzner `CAX31 Ampere ARM`.

CPU Supports `NEON` families.

Ndarray compiled with openblas installed via `libopenblas-dev`.
`OMP_NUM_THRESD=1`

```
ndarray x1024 dot                         time:   [133.54 µs 133.57 µs 133.60 µs]
ndarray x1024 cosine                      time:   [407.32 µs 407.37 µs 407.41 µs]
ndarray x1024 euclidean                   time:   [330.03 µs 330.07 µs 330.12 µs]
                                          
simsimd x1024 dot                         time:   [176.88 µs 176.89 µs 176.91 µs]
simsimd x1024 cosine                      time:   [185.93 µs 185.96 µs 185.98 µs]
simsimd x1024 euclidean                   time:   [175.84 µs 176.07 µs 176.58 µs]
                                          
f32_fallback_nofma_dot x1024              time:   [131.32 µs 131.42 µs 131.53 µs]
f32_fallback_nofma_dot xany-1301          time:   [170.16 µs 170.27 µs 170.37 µs]
f32_fallback_nofma_dot xany-1024          time:   [132.38 µs 132.53 µs 132.68 µs]
                                          
f32_fallback_nofma_cosine x1024           time:   [832.38 µs 833.28 µs 835.10 µs]
f32_fallback_nofma_cosine xany-1301       time:   [1.0590 ms 1.0591 ms 1.0592 ms]
f32_fallback_nofma_cosine xany-1024       time:   [832.13 µs 832.17 µs 832.21 µs]
                                          
f32_fallback_nofma_euclidean x1024        time:   [496.26 µs 496.29 µs 496.32 µs]
f32_fallback_nofma_euclidean xany-1301    time:   [630.57 µs 630.69 µs 630.81 µs]
f32_fallback_nofma_euclidean xany-1024    time:   [496.54 µs 496.61 µs 496.69 µs]

f32_neon_nofma_dot x1024                  time:   [90.249 µs 90.309 µs 90.369 µs]
f32_neon_nofma_dot xany-1301              time:   [116.38 µs 116.47 µs 116.57 µs]
f32_neon_nofma_dot xany-1024              time:   [88.509 µs 88.533 µs 88.559 µs]

f32_neon_nofma_cosine x1024               time:   [155.53 µs 155.54 µs 155.55 µs]
f32_neon_nofma_cosine xany-1301           time:   [403.79 µs 403.85 µs 403.93 µs]
f32_neon_nofma_cosine xany-1024           time:   [321.74 µs 321.78 µs 321.82 µs]

f32_neon_nofma_euclidean x1024            time:   [122.12 µs 122.18 µs 122.28 µs]
f32_neon_nofma_euclidean xany-1301        time:   [156.97 µs 156.99 µs 157.02 µs]
f32_neon_nofma_euclidean xany-1024        time:   [124.35 µs 124.38 µs 124.41 µs]
```