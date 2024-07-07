# ndarray-accel

A set of extension traits implementing common operations entirely in optimized SIMD 
without a dependency on blas or other C-linked libs.

This project has been extracted from some existing work on a approximate nearest 
neighbor library, and has eventually branched out to become this library.

### Not using ndarray? No problem!

This library comes in two parts, the actual extension traits for ndarray and the
low-level unsafe operations.

- `ndarray-accel` Main library targetting ndarray
- `cfavml` Accelerated vector math library with all unsafe operation, this library
  also exposes some additional functionality like the const-generic versions
  of routines that I have not yet worked out how best to utilise within the ndarray
  extension traits.
  
  * This library also only uses `core` rather than `std`.
  * This library primarily optimizes for AMD CPUs which may perform worse on Intel CPUs compared to
    the most 'optimal' Intel optimized code.
  * WARNING: This library is not for the faint of hearts, it is effectively _entirely_
    written in unsafe and makes a lot of assumptions for each routine which differ, please
    make sure to read the routine docs and not assume that just because one routine requires
    one set of rules, that all of them require the same rules and nothing else.

