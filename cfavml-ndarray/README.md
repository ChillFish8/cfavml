# cfavml-ndarray

Accelerate your ndarray operations using CFAVML's SIMD routines.

This library acts as an extension system to your existing ndarray workloads, allowing 
you to incrementally switch over to the CFAVML optimized routines brining SIMD acceleration
across all primitive integer and float types across x86 and ARM hardware.

### NOTE

Currently, this library is a WIP and requires alloc and std.