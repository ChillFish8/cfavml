### Benchmarks - AMD Ryzen 5900x (WINDOWS - no BLAS)

- **Supported Flags:** `avx2`, `fma`
- Not really a good benchmark since this is a desktop chip, your servers are going to behave very differently.

```
bench_distance_measures          fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ cosine_x1341                                │               │               │               │         │
│  ├─ cfavml_avx2_fma_f32        139.1 ns      │ 285.3 ns      │ 143.5 ns      │ 148.7 ns      │ 500     │ 1250000
│  ├─ cfavml_avx2_fma_f64        243.4 ns      │ 388.3 ns      │ 251.9 ns      │ 255.5 ns      │ 500     │ 1250000
│  ├─ cfavml_avx2_nofma_f32      118.4 ns      │ 201.3 ns      │ 122.1 ns      │ 125.1 ns      │ 500     │ 1250000
│  ├─ cfavml_avx2_nofma_f64      252.5 ns      │ 368.8 ns      │ 263.3 ns      │ 265.3 ns      │ 500     │ 1250000
│  ├─ cfavml_fallback_nofma_f32  725.9 ns      │ 1.034 µs      │ 745.3 ns      │ 753 ns        │ 500     │ 1250000
│  ├─ cfavml_fallback_nofma_f64  667.7 ns      │ 900.5 ns      │ 684.9 ns      │ 688.9 ns      │ 500     │ 1250000
│  ├─ ndarray_f32                444.1 ns      │ 567 ns        │ 452.3 ns      │ 457.5 ns      │ 500     │ 1250000
│  ├─ ndarray_f64                410 ns        │ 561.9 ns      │ 414.8 ns      │ 419.1 ns      │ 500     │ 1250000
│  ├─ simsimd_f32                859.8 ns      │ 993.3 ns      │ 869.6 ns      │ 877.1 ns      │ 500     │ 1250000
│  ╰─ simsimd_f64                862.6 ns      │ 1.007 µs      │ 872 ns        │ 876.7 ns      │ 500     │ 1250000
├─ dot_product_x1341                           │               │               │               │         │
│  ├─ cfavml_avx2_fma_f32        41.98 ns      │ 62.06 ns      │ 43.14 ns      │ 43.82 ns      │ 500     │ 1250000
│  ├─ cfavml_avx2_fma_f64        108.6 ns      │ 188.4 ns      │ 112.7 ns      │ 113.8 ns      │ 500     │ 1250000
│  ├─ cfavml_avx2_nofma_f32      57.06 ns      │ 80.34 ns      │ 58.1 ns       │ 59.16 ns      │ 500     │ 1250000
│  ├─ cfavml_avx2_nofma_f64      108.9 ns      │ 155.9 ns      │ 112.5 ns      │ 113.9 ns      │ 500     │ 1250000
│  ├─ cfavml_fallback_nofma_f32  316.5 ns      │ 468.1 ns      │ 322.2 ns      │ 325.9 ns      │ 500     │ 1250000
│  ├─ cfavml_fallback_nofma_f64  289.4 ns      │ 455 ns        │ 296.4 ns      │ 299.2 ns      │ 500     │ 1250000
│  ├─ ndarray_f32                152.7 ns      │ 222.5 ns      │ 155.1 ns      │ 156.5 ns      │ 500     │ 1250000
│  ├─ ndarray_f64                154.5 ns      │ 264.1 ns      │ 157.9 ns      │ 159.5 ns      │ 500     │ 1250000
│  ├─ simsimd_f32                838 ns        │ 958.7 ns      │ 856.2 ns      │ 856.7 ns      │ 500     │ 1250000
│  ╰─ simsimd_f64                834.7 ns      │ 893.5 ns      │ 843.7 ns      │ 846.4 ns      │ 500     │ 1250000
╰─ euclidean_x1341                             │               │               │               │         │
   ├─ cfavml_avx2_fma_f32        54.9 ns       │ 67.78 ns      │ 56.34 ns      │ 56.7 ns       │ 500     │ 1250000
   ├─ cfavml_avx2_fma_f64        97.94 ns      │ 174.5 ns      │ 100.9 ns      │ 103 ns        │ 500     │ 1250000
   ├─ cfavml_avx2_nofma_f32      55.62 ns      │ 105.1 ns      │ 57.7 ns       │ 58.78 ns      │ 500     │ 1250000
   ├─ cfavml_avx2_nofma_f64      119.7 ns      │ 160.1 ns      │ 123.1 ns      │ 123.8 ns      │ 500     │ 1250000
   ├─ cfavml_fallback_nofma_f32  317 ns        │ 532.2 ns      │ 321.3 ns      │ 323.7 ns      │ 500     │ 1250000
   ├─ cfavml_fallback_nofma_f64  353.5 ns      │ 510.7 ns      │ 362.9 ns      │ 371.8 ns      │ 500     │ 1250000
   ├─ ndarray_f32                291.6 ns      │ 485.3 ns      │ 291.9 ns      │ 295.1 ns      │ 500     │ 1250000
   ├─ ndarray_f64                399.9 ns      │ 643.7 ns      │ 412.6 ns      │ 417.3 ns      │ 500     │ 1250000
   ├─ simsimd_f32                832.3 ns      │ 979.2 ns      │ 854.1 ns      │ 858.1 ns      │ 500     │ 1250000
   ╰─ simsimd_f64                840 ns        │ 1.035 µs      │ 858.7 ns      │ 861 ns        │ 500     │ 1250000

```

### Benchmarks - Intel

Ran on a Hetzner `CX52 Intel x86`.

CPU Supports `AVX512`, `AVX2` and the `SSE` families.

Ndarray compiled with openblas installed via `libopenblas-dev`.
`OMP_NUM_THRESD=1`

Personally, I am not sure how much I would read into these results, this is an old gen XEON
and its clock accuracy changes every second... Which naturally does not give me the greatest trust in
the numbers.

##### CPU Info
```
vendor_id       : GenuineIntel
cpu family      : 6
model           : 85
model name      : Intel Xeon Processor (Skylake, IBRS, no TSX)
stepping        : 4
microcode       : 0x1
cpu MHz         : 2294.608
cache size      : 16384 KB
physical id     : 0
siblings        : 16
core id         : 15
cpu cores       : 16
```

##### Bench wo/aligned buffers
```
Timer precision: 41 ns
bench_distance_measures          fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ cosine_x1341                                │               │               │               │         │
│  ├─ cfavml_avx2_fma_f32        286.9 ns      │ 540.5 ns      │ 317.4 ns      │ 333.4 ns      │ 500     │ 1250000
│  ├─ cfavml_avx2_fma_f64        502.7 ns      │ 795.5 ns      │ 568.6 ns      │ 560.9 ns      │ 500     │ 1250000
│  ├─ cfavml_avx2_nofma_f32      341.1 ns      │ 623.7 ns      │ 395.1 ns      │ 398.9 ns      │ 500     │ 1250000
│  ├─ cfavml_avx2_nofma_f64      591.5 ns      │ 1.021 µs      │ 692.1 ns      │ 711.3 ns      │ 500     │ 1250000
│  ├─ cfavml_avx512_fma_f32      179.5 ns      │ 299.6 ns      │ 228.3 ns      │ 229.8 ns      │ 500     │ 1250000
│  ├─ cfavml_avx512_fma_f64      303.5 ns      │ 474.9 ns      │ 377.2 ns      │ 373.2 ns      │ 500     │ 1250000
│  ├─ cfavml_fallback_nofma_f32  1.334 µs      │ 2.252 µs      │ 1.573 µs      │ 1.593 µs      │ 500     │ 1250000
│  ├─ cfavml_fallback_nofma_f64  1.047 µs      │ 3.369 µs      │ 1.294 µs      │ 1.322 µs      │ 500     │ 1250000
│  ├─ ndarray_f32                12.25 µs      │ 17.55 µs      │ 14.38 µs      │ 14.31 µs      │ 500     │ 1250000
│  ├─ ndarray_f64                682.3 ns      │ 1.072 µs      │ 758.1 ns      │ 765.1 ns      │ 500     │ 1250000
│  ├─ simsimd_f32                125.5 ns      │ 187.5 ns      │ 139.9 ns      │ 144.1 ns      │ 500     │ 1250000
│  ╰─ simsimd_f64                240.6 ns      │ 347.3 ns      │ 276.5 ns      │ 286.5 ns      │ 500     │ 1250000
├─ dot_product_x1341                           │               │               │               │         │
│  ├─ cfavml_avx2_fma_f32        81.62 ns      │ 115.5 ns      │ 93.55 ns      │ 93.62 ns      │ 500     │ 1250000
│  ├─ cfavml_avx2_fma_f64        191.4 ns      │ 245.1 ns      │ 209.9 ns      │ 209.4 ns      │ 500     │ 1250000
│  ├─ cfavml_avx2_nofma_f32      100.1 ns      │ 130.2 ns      │ 109.2 ns      │ 110.2 ns      │ 500     │ 1250000
│  ├─ cfavml_avx2_nofma_f64      186.5 ns      │ 247.5 ns      │ 213.4 ns      │ 214.1 ns      │ 500     │ 1250000
│  ├─ cfavml_avx512_fma_f32      76.47 ns      │ 118.3 ns      │ 89.73 ns      │ 92.5 ns       │ 500     │ 1250000
│  ├─ cfavml_avx512_fma_f64      128.8 ns      │ 192.2 ns      │ 147 ns        │ 153.2 ns      │ 500     │ 1250000
│  ├─ cfavml_fallback_nofma_f32  542.2 ns      │ 944.4 ns      │ 627.6 ns      │ 632.9 ns      │ 500     │ 1250000
│  ├─ cfavml_fallback_nofma_f64  359.8 ns      │ 499.9 ns      │ 400.8 ns      │ 413.1 ns      │ 500     │ 1250000
│  ├─ ndarray_f32                100.9 ns      │ 147.8 ns      │ 115.3 ns      │ 118.5 ns      │ 500     │ 1250000
│  ├─ ndarray_f64                140.2 ns      │ 208.3 ns      │ 162.1 ns      │ 167.9 ns      │ 500     │ 1250000
│  ├─ simsimd_f32                101.6 ns      │ 142.4 ns      │ 115.7 ns      │ 115.7 ns      │ 500     │ 1250000
│  ╰─ simsimd_f64                225.1 ns      │ 291.6 ns      │ 256.5 ns      │ 260.9 ns      │ 500     │ 1250000
╰─ euclidean_x1341                             │               │               │               │         │
   ├─ cfavml_avx2_fma_f32        103.7 ns      │ 128.5 ns      │ 107.1 ns      │ 109.6 ns      │ 500     │ 1250000
   ├─ cfavml_avx2_fma_f64        218.3 ns      │ 257.4 ns      │ 237.7 ns      │ 233.6 ns      │ 500     │ 1250000
   ├─ cfavml_avx2_nofma_f32      134.2 ns      │ 234.4 ns      │ 149.5 ns      │ 157.6 ns      │ 500     │ 1250000
   ├─ cfavml_avx2_nofma_f64      247.4 ns      │ 438.5 ns      │ 270 ns        │ 266.1 ns      │ 500     │ 1250000
   ├─ cfavml_avx512_fma_f32      91.95 ns      │ 167.2 ns      │ 106.9 ns      │ 109.7 ns      │ 500     │ 1250000
   ├─ cfavml_avx512_fma_f64      150.2 ns      │ 201.8 ns      │ 172.9 ns      │ 174.9 ns      │ 500     │ 1250000
   ├─ cfavml_fallback_nofma_f32  659.4 ns      │ 784.8 ns      │ 666.8 ns      │ 682.6 ns      │ 500     │ 1250000
   ├─ cfavml_fallback_nofma_f64  514.6 ns      │ 630.4 ns      │ 561.8 ns      │ 567.7 ns      │ 500     │ 1250000
   ├─ ndarray_f32                3.304 µs      │ 4.387 µs      │ 3.491 µs      │ 3.514 µs      │ 500     │ 1250000
   ├─ ndarray_f64                810.4 ns      │ 912.5 ns      │ 825.8 ns      │ 836.4 ns      │ 500     │ 1250000
   ├─ simsimd_f32                135.3 ns      │ 167.9 ns      │ 138.4 ns      │ 140.4 ns      │ 500     │ 1250000
   ╰─ simsimd_f64                281.5 ns      │ 410.8 ns      │ 301.7 ns      │ 300.8 ns      │ 500     │ 1250000
```

##### Bench w/aligned buffers
```
Timer precision: 68 ns
bench_distance_measures          fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ cosine_x1341                                │               │               │               │         │
│  ├─ cfavml_avx2_fma_f32        220.9 ns      │ 525 ns        │ 252.7 ns      │ 256.1 ns      │ 500     │ 1250000
│  ├─ cfavml_avx2_fma_f64        393.3 ns      │ 513.3 ns      │ 445.5 ns      │ 443.6 ns      │ 500     │ 1250000
│  ├─ cfavml_avx2_nofma_f32      273.5 ns      │ 379 ns        │ 317.4 ns      │ 316.6 ns      │ 500     │ 1250000
│  ├─ cfavml_avx2_nofma_f64      488.3 ns      │ 667.1 ns      │ 576.3 ns      │ 574.6 ns      │ 500     │ 1250000
│  ├─ cfavml_avx512_fma_f32      142.7 ns      │ 202.3 ns      │ 159.5 ns      │ 161.4 ns      │ 500     │ 1250000
│  ├─ cfavml_avx512_fma_f64      223.4 ns      │ 320.2 ns      │ 252.1 ns      │ 254.3 ns      │ 500     │ 1250000
│  ├─ cfavml_fallback_nofma_f32  1.122 µs      │ 2.137 µs      │ 1.227 µs      │ 1.246 µs      │ 500     │ 1250000
│  ├─ cfavml_fallback_nofma_f64  931.1 ns      │ 2.286 µs      │ 1.037 µs      │ 1.045 µs      │ 500     │ 1250000
│  ├─ ndarray_f32                11.73 µs      │ 14.18 µs      │ 12.49 µs      │ 12.57 µs      │ 500     │ 1250000
│  ├─ ndarray_f64                570.6 ns      │ 796.3 ns      │ 658.4 ns      │ 670.1 ns      │ 500     │ 1250000
│  ├─ simsimd_f32                121.7 ns      │ 183.3 ns      │ 152.4 ns      │ 149.5 ns      │ 500     │ 1250000
│  ╰─ simsimd_f64                238.9 ns      │ 344.6 ns      │ 297.2 ns      │ 291.2 ns      │ 500     │ 1250000
├─ dot_product_x1341                           │               │               │               │         │
│  ├─ cfavml_avx2_fma_f32        80.11 ns      │ 117.8 ns      │ 92.16 ns      │ 92.19 ns      │ 500     │ 1250000
│  ├─ cfavml_avx2_fma_f64        131.9 ns      │ 176.3 ns      │ 147.2 ns      │ 147.3 ns      │ 500     │ 1250000
│  ├─ cfavml_avx2_nofma_f32      92.99 ns      │ 189.2 ns      │ 105.7 ns      │ 107.7 ns      │ 500     │ 1250000
│  ├─ cfavml_avx2_nofma_f64      156.5 ns      │ 209.1 ns      │ 177.7 ns      │ 178.1 ns      │ 500     │ 1250000
│  ├─ cfavml_avx512_fma_f32      59.02 ns      │ 89.5 ns       │ 65.94 ns      │ 66 ns         │ 500     │ 1250000
│  ├─ cfavml_avx512_fma_f64      82.87 ns      │ 115.6 ns      │ 92.53 ns      │ 93.54 ns      │ 500     │ 1250000
│  ├─ cfavml_fallback_nofma_f32  522.1 ns      │ 732.3 ns      │ 571.4 ns      │ 584.6 ns      │ 500     │ 1250000
│  ├─ cfavml_fallback_nofma_f64  355.1 ns      │ 501.7 ns      │ 389.2 ns      │ 398.5 ns      │ 500     │ 1250000
│  ├─ ndarray_f32                80.39 ns      │ 121.9 ns      │ 90.1 ns       │ 93.04 ns      │ 500     │ 1250000
│  ├─ ndarray_f64                94.63 ns      │ 145.3 ns      │ 105.6 ns      │ 105.5 ns      │ 500     │ 1250000
│  ├─ simsimd_f32                103.7 ns      │ 156.9 ns      │ 115.9 ns      │ 117.1 ns      │ 500     │ 1250000
│  ╰─ simsimd_f64                217.7 ns      │ 283.6 ns      │ 247.5 ns      │ 247.4 ns      │ 500     │ 1250000
╰─ euclidean_x1341                             │               │               │               │         │
   ├─ cfavml_avx2_fma_f32        89.05 ns      │ 186 ns        │ 105.6 ns      │ 108 ns        │ 500     │ 1250000
   ├─ cfavml_avx2_fma_f64        156.5 ns      │ 218.5 ns      │ 180.4 ns      │ 180.7 ns      │ 500     │ 1250000
   ├─ cfavml_avx2_nofma_f32      115.9 ns      │ 164.4 ns      │ 136.6 ns      │ 135.7 ns      │ 500     │ 1250000
   ├─ cfavml_avx2_nofma_f64      199.8 ns      │ 268.3 ns      │ 228.7 ns      │ 228.5 ns      │ 500     │ 1250000
   ├─ cfavml_avx512_fma_f32      87.33 ns      │ 124.5 ns      │ 108.6 ns      │ 109.2 ns      │ 500     │ 1250000
   ├─ cfavml_avx512_fma_f64      163.7 ns      │ 205.5 ns      │ 184.3 ns      │ 184.4 ns      │ 500     │ 1250000
   ├─ cfavml_fallback_nofma_f32  672.2 ns      │ 806.5 ns      │ 738 ns        │ 737.8 ns      │ 500     │ 1250000
   ├─ cfavml_fallback_nofma_f64  582.3 ns      │ 987.3 ns      │ 633.5 ns      │ 622.2 ns      │ 500     │ 1250000
   ├─ ndarray_f32                3.744 µs      │ 4.146 µs      │ 3.832 µs      │ 3.84 µs       │ 500     │ 1250000
   ├─ ndarray_f64                694.7 ns      │ 1.141 µs      │ 739.4 ns      │ 742 ns        │ 500     │ 1250000
   ├─ simsimd_f32                135.7 ns      │ 172.1 ns      │ 149.5 ns      │ 151.3 ns      │ 500     │ 1250000
   ╰─ simsimd_f64                284.3 ns      │ 347.8 ns      │ 312.4 ns      │ 313.4 ns      │ 500     │ 1250000
```

### Benchmarks - AMD

Ran on a Hetzner `CPX51 AMD x86`.

CPU Supports `AVX2` and the `SSE` families.

Ndarray compiled with openblas installed via `libopenblas-dev`.
`OMP_NUM_THRESD=1`

##### CPU Info
```
vendor_id       : AuthenticAMD
cpu family      : 23
model           : 49
model name      : AMD EPYC Processor
stepping        : 0
microcode       : 0x1000065
cpu MHz         : 2445.406
cache size      : 512 KB
physical id     : 0
siblings        : 16
core id         : 15
cpu cores       : 16
```

##### Bench wo/aligned buffers
```
Timer precision: 29 ns
bench_distance_measures          fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ cosine_x1341                                │               │               │               │         │
│  ├─ cfavml_avx2_fma_f32        181.5 ns      │ 240.3 ns      │ 187.8 ns      │ 190.8 ns      │ 500     │ 1250000
│  ├─ cfavml_avx2_fma_f64        345.2 ns      │ 429.8 ns      │ 351.6 ns      │ 356.5 ns      │ 500     │ 1250000
│  ├─ cfavml_avx2_nofma_f32      188.8 ns      │ 265.9 ns      │ 195.6 ns      │ 197.5 ns      │ 500     │ 1250000
│  ├─ cfavml_avx2_nofma_f64      352.4 ns      │ 571 ns        │ 363.9 ns      │ 374.4 ns      │ 500     │ 1250000
│  ├─ cfavml_fallback_nofma_f32  1.038 µs      │ 1.532 µs      │ 1.132 µs      │ 1.129 µs      │ 500     │ 1250000
│  ├─ cfavml_fallback_nofma_f64  968.8 ns      │ 1.254 µs      │ 990.6 ns      │ 1.028 µs      │ 500     │ 1250000
│  ├─ ndarray_f32                488.1 ns      │ 789.6 ns      │ 492.5 ns      │ 494.7 ns      │ 500     │ 1250000
│  ├─ ndarray_f64                523.9 ns      │ 588.7 ns      │ 542.4 ns      │ 544.1 ns      │ 500     │ 1250000
│  ├─ simsimd_f32                1.253 µs      │ 1.329 µs      │ 1.26 µs       │ 1.261 µs      │ 500     │ 1250000
│  ╰─ simsimd_f64                1.256 µs      │ 1.31 µs       │ 1.263 µs      │ 1.265 µs      │ 500     │ 1250000
├─ dot_product_x1341                           │               │               │               │         │
│  ├─ cfavml_avx2_fma_f32        59.48 ns      │ 78.08 ns      │ 60.42 ns      │ 62.19 ns      │ 500     │ 1250000
│  ├─ cfavml_avx2_fma_f64        151 ns        │ 178 ns        │ 152.6 ns      │ 153.3 ns      │ 500     │ 1250000
│  ├─ cfavml_avx2_nofma_f32      63.79 ns      │ 73.85 ns      │ 64.87 ns      │ 65.8 ns       │ 500     │ 1250000
│  ├─ cfavml_avx2_nofma_f64      154.4 ns      │ 167.7 ns      │ 157.8 ns      │ 157.7 ns      │ 500     │ 1250000
│  ├─ cfavml_fallback_nofma_f32  457.9 ns      │ 548.8 ns      │ 462.3 ns      │ 463.9 ns      │ 500     │ 1250000
│  ├─ cfavml_fallback_nofma_f64  422.8 ns      │ 500.4 ns      │ 430.5 ns      │ 431.5 ns      │ 500     │ 1250000
│  ├─ ndarray_f32                78.32 ns      │ 96.13 ns      │ 79.69 ns      │ 80.84 ns      │ 500     │ 1250000
│  ├─ ndarray_f64                165.6 ns      │ 204.6 ns      │ 168 ns        │ 168.4 ns      │ 500     │ 1250000
│  ├─ simsimd_f32                1.213 µs      │ 1.299 µs      │ 1.226 µs      │ 1.229 µs      │ 500     │ 1250000
│  ╰─ simsimd_f64                1.221 µs      │ 1.26 µs       │ 1.228 µs      │ 1.23 µs       │ 500     │ 1250000
╰─ euclidean_x1341                             │               │               │               │         │
   ├─ cfavml_avx2_fma_f32        68.2 ns       │ 101.6 ns      │ 71.45 ns      │ 72.8 ns       │ 500     │ 1250000
   ├─ cfavml_avx2_fma_f64        139.5 ns      │ 191.7 ns      │ 144.6 ns      │ 148.6 ns      │ 500     │ 1250000
   ├─ cfavml_avx2_nofma_f32      70.91 ns      │ 83.91 ns      │ 72.86 ns      │ 73.47 ns      │ 500     │ 1250000
   ├─ cfavml_avx2_nofma_f64      141.8 ns      │ 223 ns        │ 147.6 ns      │ 155 ns        │ 500     │ 1250000
   ├─ cfavml_fallback_nofma_f32  468.1 ns      │ 642.4 ns      │ 480.2 ns      │ 489.4 ns      │ 500     │ 1250000
   ├─ cfavml_fallback_nofma_f64  475.3 ns      │ 618.7 ns      │ 482 ns        │ 490.9 ns      │ 500     │ 1250000
   ├─ ndarray_f32                385.6 ns      │ 460.3 ns      │ 393 ns        │ 395.3 ns      │ 500     │ 1250000
   ├─ ndarray_f64                513.1 ns      │ 664.6 ns      │ 520.2 ns      │ 524.9 ns      │ 500     │ 1250000
   ├─ simsimd_f32                1.219 µs      │ 1.277 µs      │ 1.226 µs      │ 1.228 µs      │ 500     │ 1250000
   ╰─ simsimd_f64                1.228 µs      │ 1.276 µs      │ 1.236 µs      │ 1.238 µs      │ 500     │ 1250000
```

##### Bench w/aligned buffers
```
Timer precision: 29 ns
bench_distance_measures          fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ cosine_x1341                                │               │               │               │         │
│  ├─ cfavml_avx2_fma_f32        170 ns        │ 256.4 ns      │ 175.3 ns      │ 177.7 ns      │ 500     │ 1250000
│  ├─ cfavml_avx2_fma_f64        309.6 ns      │ 486.9 ns      │ 324.4 ns      │ 337.1 ns      │ 500     │ 1250000
│  ├─ cfavml_avx2_nofma_f32      169.8 ns      │ 263.7 ns      │ 174.5 ns      │ 177.7 ns      │ 500     │ 1250000
│  ├─ cfavml_avx2_nofma_f64      303.3 ns      │ 454.3 ns      │ 314.5 ns      │ 324.4 ns      │ 500     │ 1250000
│  ├─ cfavml_fallback_nofma_f32  1.036 µs      │ 1.374 µs      │ 1.048 µs      │ 1.063 µs      │ 500     │ 1250000
│  ├─ cfavml_fallback_nofma_f64  975.5 ns      │ 1.134 µs      │ 993.4 ns      │ 1.002 µs      │ 500     │ 1250000
│  ├─ ndarray_f32                488 ns        │ 544.3 ns      │ 493.9 ns      │ 496.1 ns      │ 500     │ 1250000
│  ├─ ndarray_f64                468.2 ns      │ 603.7 ns      │ 475.8 ns      │ 491.6 ns      │ 500     │ 1250000
│  ├─ simsimd_f32                1.254 µs      │ 1.29 µs       │ 1.261 µs      │ 1.262 µs      │ 500     │ 1250000
│  ╰─ simsimd_f64                1.256 µs      │ 1.337 µs      │ 1.267 µs      │ 1.272 µs      │ 500     │ 1250000
├─ dot_product_x1341                           │               │               │               │         │
│  ├─ cfavml_avx2_fma_f32        59.55 ns      │ 77.42 ns      │ 60.45 ns      │ 61.7 ns       │ 500     │ 1250000
│  ├─ cfavml_avx2_fma_f64        106.3 ns      │ 128.3 ns      │ 107.3 ns      │ 109.1 ns      │ 500     │ 1250000
│  ├─ cfavml_avx2_nofma_f32      63.83 ns      │ 79.29 ns      │ 64.52 ns      │ 65.47 ns      │ 500     │ 1250000
│  ├─ cfavml_avx2_nofma_f64      109.6 ns      │ 134.8 ns      │ 110.5 ns      │ 112.2 ns      │ 500     │ 1250000
│  ├─ cfavml_fallback_nofma_f32  457.2 ns      │ 555.1 ns      │ 463.3 ns      │ 465.7 ns      │ 500     │ 1250000
│  ├─ cfavml_fallback_nofma_f64  423.1 ns      │ 475.8 ns      │ 434.6 ns      │ 435.9 ns      │ 500     │ 1250000
│  ├─ ndarray_f32                78.39 ns      │ 115.2 ns      │ 79.73 ns      │ 81 ns         │ 500     │ 1250000
│  ├─ ndarray_f64                118.2 ns      │ 172.4 ns      │ 120.3 ns      │ 121.9 ns      │ 500     │ 1250000
│  ├─ simsimd_f32                1.214 µs      │ 1.35 µs       │ 1.235 µs      │ 1.241 µs      │ 500     │ 1250000
│  ╰─ simsimd_f64                1.222 µs      │ 1.306 µs      │ 1.23 µs       │ 1.232 µs      │ 500     │ 1250000
╰─ euclidean_x1341                             │               │               │               │         │
   ├─ cfavml_avx2_fma_f32        67.9 ns       │ 109.2 ns      │ 69.65 ns      │ 71.24 ns      │ 500     │ 1250000
   ├─ cfavml_avx2_fma_f64        113.9 ns      │ 168.7 ns      │ 119 ns        │ 123 ns        │ 500     │ 1250000
   ├─ cfavml_avx2_nofma_f32      71.16 ns      │ 103.2 ns      │ 74.89 ns      │ 76.91 ns      │ 500     │ 1250000
   ├─ cfavml_avx2_nofma_f64      118.2 ns      │ 178.9 ns      │ 122.4 ns      │ 125.3 ns      │ 500     │ 1250000
   ├─ cfavml_fallback_nofma_f32  464.8 ns      │ 725.6 ns      │ 481.7 ns      │ 504.3 ns      │ 500     │ 1250000
   ├─ cfavml_fallback_nofma_f64  473.6 ns      │ 674.7 ns      │ 481.1 ns      │ 489.1 ns      │ 500     │ 1250000
   ├─ ndarray_f32                401.1 ns      │ 529.1 ns      │ 410.7 ns      │ 417.6 ns      │ 500     │ 1250000
   ├─ ndarray_f64                462.9 ns      │ 728 ns        │ 474.8 ns      │ 484.1 ns      │ 500     │ 1250000
   ├─ simsimd_f32                1.218 µs      │ 1.34 µs       │ 1.228 µs      │ 1.236 µs      │ 500     │ 1250000
   ╰─ simsimd_f64                1.228 µs      │ 1.321 µs      │ 1.24 µs       │ 1.244 µs      │ 500     │ 1250000
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
