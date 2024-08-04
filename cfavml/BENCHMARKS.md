### Benchmarks - AMD Ryzen 5900x 

- **Supported Flags:** `avx2`, `fma`
- Not really a good benchmark since this is a desktop chip, your servers are going to behave very differently.

```
Timer precision: 20 ns
bench_distance_measures  fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ cosine_x1341                        │               │               │               │         │
│  ├─ cfavml_f32         132.9 ns      │ 288.3 ns      │ 135.7 ns      │ 149.8 ns      │ 500     │ 1250000
│  ├─ cfavml_f64         244.5 ns      │ 318.9 ns      │ 251.7 ns      │ 252.7 ns      │ 500     │ 1250000
│  ├─ ndarray_f32        345.2 ns      │ 501.4 ns      │ 359.7 ns      │ 360.2 ns      │ 500     │ 1250000
│  ├─ ndarray_f64        362.1 ns      │ 438.9 ns      │ 382.9 ns      │ 381.9 ns      │ 500     │ 1250000
│  ├─ simsimd_f32        834.5 ns      │ 1.033 µs      │ 862.6 ns      │ 866.8 ns      │ 500     │ 1250000
│  ╰─ simsimd_f64        852.2 ns      │ 898.3 ns      │ 857.6 ns      │ 860.1 ns      │ 500     │ 1250000
├─ dot_product_x1341                   │               │               │               │         │
│  ├─ cfavml_f32         51.54 ns      │ 55.62 ns      │ 51.89 ns      │ 52.16 ns      │ 500     │ 1250000
│  ├─ cfavml_f64         105.6 ns      │ 109.1 ns      │ 105.6 ns      │ 106.2 ns      │ 500     │ 1250000
│  ├─ ndarray_f32        69.26 ns      │ 79.68 ns      │ 70.24 ns      │ 70.57 ns      │ 500     │ 1250000
│  ├─ ndarray_f64        112.8 ns      │ 119.3 ns      │ 113.6 ns      │ 114.1 ns      │ 500     │ 1250000
│  ├─ simsimd_f32        806.4 ns      │ 846 ns        │ 809.7 ns      │ 813 ns        │ 500     │ 1250000
│  ╰─ simsimd_f64        813.3 ns      │ 860.4 ns      │ 820.2 ns      │ 824 ns        │ 500     │ 1250000
╰─ euclidean_x1341                     │               │               │               │         │
   ├─ cfavml_f32         52.2 ns       │ 57.9 ns       │ 55.24 ns      │ 55.23 ns      │ 500     │ 1250000
   ├─ cfavml_f64         96.99 ns      │ 104.1 ns      │ 99.12 ns      │ 99.31 ns      │ 500     │ 1250000
   ├─ ndarray_f32        263.4 ns      │ 281.6 ns      │ 267.3 ns      │ 267.9 ns      │ 500     │ 1250000
   ├─ ndarray_f64        364 ns        │ 380.3 ns      │ 367.3 ns      │ 368 ns        │ 500     │ 1250000
   ├─ simsimd_f32        809.7 ns      │ 854.5 ns      │ 814.1 ns      │ 818 ns        │ 500     │ 1250000
   ╰─ simsimd_f64        819.4 ns      │ 1.011 µs      │ 856.6 ns      │ 852 ns        │ 500     │ 1250000

     Running benches/bench_min_max_sum.rs (target/release/deps/bench_min_max_sum-5a6f9afed7ede9a3)
Timer precision: 10 ns
bench_min_max_sum      fastest       │ slowest       │ median        │ mean          │ samples │ iters
├─ op_max                            │               │               │               │         │
│  ├─ f32_max_cfavml   60.08 ns      │ 64.53 ns      │ 62.03 ns      │ 62.16 ns      │ 500     │ 5000000
│  │                   72.27 Gitem/s │ 67.28 Gitem/s │ 69.99 Gitem/s │ 69.84 Gitem/s │         │
│  ├─ f32_max_ndarray  3.211 µs      │ 3.505 µs      │ 3.288 µs      │ 3.302 µs      │ 500     │ 5000000
│  │                   1.352 Gitem/s │ 1.238 Gitem/s │ 1.32 Gitem/s  │ 1.314 Gitem/s │         │
│  ├─ f64_max_cfavml   171.3 ns      │ 179.1 ns      │ 173.3 ns      │ 173.4 ns      │ 500     │ 5000000
│  │                   25.34 Gitem/s │ 24.23 Gitem/s │ 25.04 Gitem/s │ 25.03 Gitem/s │         │
│  ├─ f64_max_ndarray  3.269 µs      │ 3.372 µs      │ 3.287 µs      │ 3.29 µs       │ 500     │ 5000000
│  │                   1.327 Gitem/s │ 1.287 Gitem/s │ 1.32 Gitem/s  │ 1.319 Gitem/s │         │
│  ├─ i16_max_cfavml   32.43 ns      │ 47.44 ns      │ 32.95 ns      │ 33.06 ns      │ 500     │ 5000000
│  │                   133.8 Gitem/s │ 91.51 Gitem/s │ 131.7 Gitem/s │ 131.3 Gitem/s │         │
│  ├─ i16_max_ndarray  68.54 ns      │ 134 ns        │ 68.85 ns      │ 69.5 ns       │ 500     │ 5000000
│  │                   63.34 Gitem/s │ 32.38 Gitem/s │ 63.06 Gitem/s │ 62.47 Gitem/s │         │
│  ├─ i32_max_cfavml   61.84 ns      │ 67.17 ns      │ 62.41 ns      │ 62.5 ns       │ 500     │ 5000000
│  │                   70.21 Gitem/s │ 64.63 Gitem/s │ 69.56 Gitem/s │ 69.46 Gitem/s │         │
│  ├─ i32_max_ndarray  444.5 ns      │ 576.3 ns      │ 447.2 ns      │ 449.4 ns      │ 500     │ 5000000
│  │                   9.766 Gitem/s │ 7.533 Gitem/s │ 9.708 Gitem/s │ 9.659 Gitem/s │         │
│  ├─ i64_max_cfavml   204.7 ns      │ 235.2 ns      │ 207.2 ns      │ 207.7 ns      │ 500     │ 5000000
│  │                   21.2 Gitem/s  │ 18.46 Gitem/s │ 20.95 Gitem/s │ 20.9 Gitem/s  │         │
│  ├─ i64_max_ndarray  2.303 µs      │ 2.822 µs      │ 2.345 µs      │ 2.344 µs      │ 500     │ 5000000
│  │                   1.885 Gitem/s │ 1.538 Gitem/s │ 1.851 Gitem/s │ 1.852 Gitem/s │         │
│  ├─ i8_max_cfavml    18.98 ns      │ 24.04 ns      │ 19.3 ns       │ 19.35 ns      │ 500     │ 5000000
│  │                   228.7 Gitem/s │ 180.5 Gitem/s │ 224.9 Gitem/s │ 224.3 Gitem/s │         │
│  ├─ i8_max_ndarray   114.8 ns      │ 175.7 ns      │ 117.6 ns      │ 118.5 ns      │ 500     │ 5000000
│  │                   37.79 Gitem/s │ 24.7 Gitem/s  │ 36.91 Gitem/s │ 36.63 Gitem/s │         │
│  ├─ u16_max_cfavml   32.47 ns      │ 45.21 ns      │ 33.19 ns      │ 33.28 ns      │ 500     │ 5000000
│  │                   133.7 Gitem/s │ 96.02 Gitem/s │ 130.8 Gitem/s │ 130.4 Gitem/s │         │
│  ├─ u16_max_ndarray  126.1 ns      │ 134 ns        │ 127.4 ns      │ 127.5 ns      │ 500     │ 5000000
│  │                   34.41 Gitem/s │ 32.38 Gitem/s │ 34.06 Gitem/s │ 34.03 Gitem/s │         │
│  ├─ u32_max_cfavml   62.42 ns      │ 67.99 ns      │ 63.3 ns       │ 63.46 ns      │ 500     │ 5000000
│  │                   69.55 Gitem/s │ 63.85 Gitem/s │ 68.58 Gitem/s │ 68.41 Gitem/s │         │
│  ├─ u32_max_ndarray  581.5 ns      │ 603 ns        │ 593.2 ns      │ 593.3 ns      │ 500     │ 5000000
│  │                   7.465 Gitem/s │ 7.2 Gitem/s   │ 7.318 Gitem/s │ 7.317 Gitem/s │         │
│  ├─ u64_max_cfavml   286.6 ns      │ 306.9 ns      │ 292.3 ns      │ 292.6 ns      │ 500     │ 5000000
│  │                   15.14 Gitem/s │ 14.14 Gitem/s │ 14.85 Gitem/s │ 14.83 Gitem/s │         │
│  ├─ u64_max_ndarray  2.31 µs       │ 2.484 µs      │ 2.353 µs      │ 2.353 µs      │ 500     │ 5000000
│  │                   1.878 Gitem/s │ 1.747 Gitem/s │ 1.845 Gitem/s │ 1.845 Gitem/s │         │
│  ├─ u8_max_cfavml    18.55 ns      │ 32.24 ns      │ 18.94 ns      │ 19.45 ns      │ 500     │ 5000000
│  │                   234 Gitem/s   │ 134.6 Gitem/s │ 229.2 Gitem/s │ 223.2 Gitem/s │         │
│  ╰─ u8_max_ndarray   42.38 ns      │ 81.5 ns       │ 43.22 ns      │ 43.93 ns      │ 500     │ 5000000
│                      102.4 Gitem/s │ 53.27 Gitem/s │ 100.4 Gitem/s │ 98.82 Gitem/s │         │
├─ op_min                            │               │               │               │         │
│  ├─ f32_min_cfavml   61.41 ns      │ 76.65 ns      │ 62.59 ns      │ 62.64 ns      │ 500     │ 5000000
│  │                   70.69 Gitem/s │ 56.64 Gitem/s │ 69.36 Gitem/s │ 69.31 Gitem/s │         │
│  ├─ f32_min_ndarray  3.263 µs      │ 3.456 µs      │ 3.34 µs       │ 3.33 µs       │ 500     │ 5000000
│  │                   1.33 Gitem/s  │ 1.256 Gitem/s │ 1.299 Gitem/s │ 1.303 Gitem/s │         │
│  ├─ f64_min_cfavml   163.3 ns      │ 258.3 ns      │ 167.8 ns      │ 168.9 ns      │ 500     │ 5000000
│  │                   26.58 Gitem/s │ 16.8 Gitem/s  │ 25.86 Gitem/s │ 25.69 Gitem/s │         │
│  ├─ f64_min_ndarray  3.266 µs      │ 3.465 µs      │ 3.34 µs       │ 3.334 µs      │ 500     │ 5000000
│  │                   1.329 Gitem/s │ 1.253 Gitem/s │ 1.299 Gitem/s │ 1.301 Gitem/s │         │
│  ├─ i16_min_cfavml   32.39 ns      │ 54.96 ns      │ 33.31 ns      │ 35.05 ns      │ 500     │ 5000000
│  │                   134 Gitem/s   │ 78.99 Gitem/s │ 130.3 Gitem/s │ 123.8 Gitem/s │         │
│  ├─ i16_min_ndarray  69.95 ns      │ 74.06 ns      │ 70.6 ns       │ 70.66 ns      │ 500     │ 5000000
│  │                   62.07 Gitem/s │ 58.62 Gitem/s │ 61.49 Gitem/s │ 61.44 Gitem/s │         │
│  ├─ i32_min_cfavml   62.98 ns      │ 102.4 ns      │ 63.79 ns      │ 65.86 ns      │ 500     │ 5000000
│  │                   68.94 Gitem/s │ 42.39 Gitem/s │ 68.05 Gitem/s │ 65.91 Gitem/s │         │
│  ├─ i32_min_ndarray  442.8 ns      │ 555.2 ns      │ 454.3 ns      │ 456 ns        │ 500     │ 5000000
│  │                   9.804 Gitem/s │ 7.82 Gitem/s  │ 9.557 Gitem/s │ 9.52 Gitem/s  │         │
│  ├─ i64_min_cfavml   202.4 ns      │ 289 ns        │ 207.5 ns      │ 210.5 ns      │ 500     │ 5000000
│  │                   21.44 Gitem/s │ 15.02 Gitem/s │ 20.91 Gitem/s │ 20.61 Gitem/s │         │
│  ├─ i64_min_ndarray  2.316 µs      │ 2.514 µs      │ 2.353 µs      │ 2.351 µs      │ 500     │ 5000000
│  │                   1.874 Gitem/s │ 1.726 Gitem/s │ 1.844 Gitem/s │ 1.846 Gitem/s │         │
│  ├─ i8_min_cfavml    18.88 ns      │ 22.22 ns      │ 19.36 ns      │ 19.44 ns      │ 500     │ 5000000
│  │                   229.9 Gitem/s │ 195.3 Gitem/s │ 224.1 Gitem/s │ 223.3 Gitem/s │         │
│  ├─ i8_min_ndarray   112.9 ns      │ 126.5 ns      │ 115.6 ns      │ 116.1 ns      │ 500     │ 5000000
│  │                   38.45 Gitem/s │ 34.31 Gitem/s │ 37.55 Gitem/s │ 37.37 Gitem/s │         │
│  ├─ u16_min_cfavml   32.53 ns      │ 39.85 ns      │ 33.02 ns      │ 33.32 ns      │ 500     │ 5000000
│  │                   133.4 Gitem/s │ 108.9 Gitem/s │ 131.4 Gitem/s │ 130.2 Gitem/s │         │
│  ├─ u16_min_ndarray  125.3 ns      │ 140.3 ns      │ 127.2 ns      │ 127.7 ns      │ 500     │ 5000000
│  │                   34.62 Gitem/s │ 30.93 Gitem/s │ 34.12 Gitem/s │ 33.99 Gitem/s │         │
│  ├─ u32_min_cfavml   62.14 ns      │ 67.16 ns      │ 62.94 ns      │ 63.08 ns      │ 500     │ 5000000
│  │                   69.87 Gitem/s │ 64.65 Gitem/s │ 68.98 Gitem/s │ 68.82 Gitem/s │         │
│  ├─ u32_min_ndarray  586.4 ns      │ 673 ns        │ 595.1 ns      │ 596 ns        │ 500     │ 5000000
│  │                   7.403 Gitem/s │ 6.451 Gitem/s │ 7.295 Gitem/s │ 7.284 Gitem/s │         │
│  ├─ u64_min_cfavml   289.6 ns      │ 307.7 ns      │ 293.4 ns      │ 294.3 ns      │ 500     │ 5000000
│  │                   14.99 Gitem/s │ 14.1 Gitem/s  │ 14.79 Gitem/s │ 14.75 Gitem/s │         │
│  ├─ u64_min_ndarray  2.288 µs      │ 2.436 µs      │ 2.344 µs      │ 2.34 µs       │ 500     │ 5000000
│  │                   1.896 Gitem/s │ 1.781 Gitem/s │ 1.852 Gitem/s │ 1.855 Gitem/s │         │
│  ├─ u8_min_cfavml    18.36 ns      │ 20.11 ns      │ 18.78 ns      │ 18.79 ns      │ 500     │ 5000000
│  │                   236.4 Gitem/s │ 215.8 Gitem/s │ 231.1 Gitem/s │ 231 Gitem/s   │         │
│  ╰─ u8_min_ndarray   41.47 ns      │ 91.79 ns      │ 42.54 ns      │ 42.85 ns      │ 500     │ 5000000
│                      104.6 Gitem/s │ 47.3 Gitem/s  │ 102 Gitem/s   │ 101.3 Gitem/s │         │
╰─ op_sum                            │               │               │               │         │
   ├─ f32_sum_cfavml   63.27 ns      │ 72.34 ns      │ 64 ns         │ 64.01 ns      │ 500     │ 5000000
   │                   68.61 Gitem/s │ 60.02 Gitem/s │ 67.83 Gitem/s │ 67.83 Gitem/s │         │
   ├─ f32_sum_naive    2.711 µs      │ 2.831 µs      │ 2.769 µs      │ 2.768 µs      │ 500     │ 5000000
   │                   1.601 Gitem/s │ 1.533 Gitem/s │ 1.567 Gitem/s │ 1.568 Gitem/s │         │
   ├─ f32_sum_ndarray  374.9 ns      │ 390.5 ns      │ 378.9 ns      │ 378.9 ns      │ 500     │ 5000000
   │                   11.58 Gitem/s │ 11.11 Gitem/s │ 11.45 Gitem/s │ 11.45 Gitem/s │         │
   ├─ f64_sum_cfavml   161.7 ns      │ 280.1 ns      │ 165.3 ns      │ 166.8 ns      │ 500     │ 5000000
   │                   26.85 Gitem/s │ 15.49 Gitem/s │ 26.26 Gitem/s │ 26.02 Gitem/s │         │
   ├─ f64_sum_naive    2.694 µs      │ 2.863 µs      │ 2.767 µs      │ 2.76 µs       │ 500     │ 5000000
   │                   1.611 Gitem/s │ 1.516 Gitem/s │ 1.568 Gitem/s │ 1.572 Gitem/s │         │
   ├─ f64_sum_ndarray  371.8 ns      │ 378.5 ns      │ 375.5 ns      │ 375.6 ns      │ 500     │ 5000000
   │                   11.67 Gitem/s │ 11.46 Gitem/s │ 11.56 Gitem/s │ 11.55 Gitem/s │         │
   ├─ i16_sum_cfavml   32.57 ns      │ 35.34 ns      │ 33.18 ns      │ 33.2 ns       │ 500     │ 5000000
   │                   133.2 Gitem/s │ 122.8 Gitem/s │ 130.8 Gitem/s │ 130.7 Gitem/s │         │
   ├─ i16_sum_naive    65.75 ns      │ 79.22 ns      │ 66.38 ns      │ 66.51 ns      │ 500     │ 5000000
   │                   66.03 Gitem/s │ 54.8 Gitem/s  │ 65.41 Gitem/s │ 65.28 Gitem/s │         │
   ├─ i16_sum_ndarray  67.04 ns      │ 74.09 ns      │ 70.99 ns      │ 70.8 ns       │ 500     │ 5000000
   │                   64.76 Gitem/s │ 58.6 Gitem/s  │ 61.15 Gitem/s │ 61.32 Gitem/s │         │
   ├─ i32_sum_cfavml   63.32 ns      │ 66.14 ns      │ 63.88 ns      │ 63.94 ns      │ 500     │ 5000000
   │                   68.56 Gitem/s │ 65.64 Gitem/s │ 67.96 Gitem/s │ 67.9 Gitem/s  │         │
   ├─ i32_sum_naive    125.8 ns      │ 246.1 ns      │ 126.7 ns      │ 127.9 ns      │ 500     │ 5000000
   │                   34.5 Gitem/s  │ 17.64 Gitem/s │ 34.26 Gitem/s │ 33.92 Gitem/s │         │
   ├─ i32_sum_ndarray  366.3 ns      │ 491.1 ns      │ 372.3 ns      │ 373 ns        │ 500     │ 5000000
   │                   11.85 Gitem/s │ 8.84 Gitem/s  │ 11.65 Gitem/s │ 11.63 Gitem/s │         │
   ├─ i64_sum_cfavml   167 ns        │ 207 ns        │ 170.9 ns      │ 172 ns        │ 500     │ 5000000
   │                   25.99 Gitem/s │ 20.97 Gitem/s │ 25.39 Gitem/s │ 25.24 Gitem/s │         │
   ├─ i64_sum_naive    485.1 ns      │ 607.7 ns      │ 487.1 ns      │ 488.9 ns      │ 500     │ 5000000
   │                   8.949 Gitem/s │ 7.144 Gitem/s │ 8.913 Gitem/s │ 8.88 Gitem/s  │         │
   ├─ i64_sum_ndarray  420.9 ns      │ 458.3 ns      │ 426.4 ns      │ 432 ns        │ 500     │ 5000000
   │                   10.31 Gitem/s │ 9.472 Gitem/s │ 10.18 Gitem/s │ 10.05 Gitem/s │         │
   ├─ i8_sum_cfavml    18.58 ns      │ 27.53 ns      │ 18.71 ns      │ 18.8 ns       │ 500     │ 5000000
   │                   233.6 Gitem/s │ 157.7 Gitem/s │ 232 Gitem/s   │ 230.8 Gitem/s │         │
   ├─ i8_sum_naive     64.78 ns      │ 67.93 ns      │ 65.14 ns      │ 65.33 ns      │ 500     │ 5000000
   │                   67.02 Gitem/s │ 63.91 Gitem/s │ 66.65 Gitem/s │ 66.45 Gitem/s │         │
   ├─ i8_sum_ndarray   366.1 ns      │ 373.3 ns      │ 367.6 ns      │ 368 ns        │ 500     │ 5000000
   │                   11.85 Gitem/s │ 11.63 Gitem/s │ 11.8 Gitem/s  │ 11.79 Gitem/s │         │
   ├─ u16_sum_cfavml   32.39 ns      │ 34.43 ns      │ 32.85 ns      │ 32.87 ns      │ 500     │ 5000000
   │                   134 Gitem/s   │ 126 Gitem/s   │ 132.1 Gitem/s │ 132 Gitem/s   │         │
   ├─ u16_sum_naive    65.08 ns      │ 68.36 ns      │ 65.84 ns      │ 65.95 ns      │ 500     │ 5000000
   │                   66.71 Gitem/s │ 63.51 Gitem/s │ 65.94 Gitem/s │ 65.82 Gitem/s │         │
   ├─ u16_sum_ndarray  66.89 ns      │ 81.23 ns      │ 70.42 ns      │ 70.27 ns      │ 500     │ 5000000
   │                   64.9 Gitem/s  │ 53.45 Gitem/s │ 61.65 Gitem/s │ 61.78 Gitem/s │         │
   ├─ u32_sum_cfavml   62.76 ns      │ 64.96 ns      │ 63.21 ns      │ 63.34 ns      │ 500     │ 5000000
   │                   69.17 Gitem/s │ 66.83 Gitem/s │ 68.68 Gitem/s │ 68.54 Gitem/s │         │
   ├─ u32_sum_naive    124.1 ns      │ 137.7 ns      │ 125.5 ns      │ 125.7 ns      │ 500     │ 5000000
   │                   34.97 Gitem/s │ 31.51 Gitem/s │ 34.59 Gitem/s │ 34.53 Gitem/s │         │
   ├─ u32_sum_ndarray  364.1 ns      │ 398.7 ns      │ 367.8 ns      │ 368 ns        │ 500     │ 5000000
   │                   11.92 Gitem/s │ 10.88 Gitem/s │ 11.8 Gitem/s  │ 11.79 Gitem/s │         │
   ├─ u64_sum_cfavml   166.5 ns      │ 184 ns        │ 168.4 ns      │ 169.5 ns      │ 500     │ 5000000
   │                   26.07 Gitem/s │ 23.59 Gitem/s │ 25.77 Gitem/s │ 25.6 Gitem/s  │         │
   ├─ u64_sum_naive    477.2 ns      │ 489.8 ns      │ 478.8 ns      │ 479 ns        │ 500     │ 5000000
   │                   9.098 Gitem/s │ 8.863 Gitem/s │ 9.068 Gitem/s │ 9.064 Gitem/s │         │
   ├─ u64_sum_ndarray  418.1 ns      │ 457.4 ns      │ 420.7 ns      │ 425.3 ns      │ 500     │ 5000000
   │                   10.38 Gitem/s │ 9.492 Gitem/s │ 10.31 Gitem/s │ 10.2 Gitem/s  │         │
   ├─ u8_sum_cfavml    18.38 ns      │ 19.89 ns      │ 18.48 ns      │ 18.51 ns      │ 500     │ 5000000
   │                   236.2 Gitem/s │ 218.2 Gitem/s │ 234.9 Gitem/s │ 234.5 Gitem/s │         │
   ├─ u8_sum_naive     64.43 ns      │ 66.17 ns      │ 64.65 ns      │ 64.83 ns      │ 500     │ 5000000
   │                   67.38 Gitem/s │ 65.61 Gitem/s │ 67.15 Gitem/s │ 66.97 Gitem/s │         │
   ╰─ u8_sum_ndarray   363.3 ns      │ 378.2 ns      │ 364.5 ns      │ 365 ns        │ 500     │ 5000000
                       11.95 Gitem/s │ 11.47 Gitem/s │ 11.9 Gitem/s  │ 11.89 Gitem/s │         │
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
