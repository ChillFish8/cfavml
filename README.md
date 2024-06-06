# ndarray-accel

A set of extension traits implementing common operations entirely in optimized SIMD 
without a dependency on blas or other C-linked libs.

This project has been extracted from some existing work on a approximate nearest 
neighbor library, and has eventually branched out to become this library.

### Not using ndarray? No problem!

This library comes in two parts, the actual extension traits for ndarray and the
low-level unsafe operations.


