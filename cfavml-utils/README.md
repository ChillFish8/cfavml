# CFAVML Utils

> _Common utilities for maximizing performance during computation with `cfavml`_

This library primarily revolves around thread pools and thread management, 

## Features

##### `env-var-compat`
Enables common env var compatibility.

This enables CFAVML to use common env vars like `OMP_NUM_THREADS` or `OPENBLAS_NUM_THREADS`
to configure the CPU limits of the system.

## Notes on Threading

This library manages threadpools, all threads a pinned to specific cores
in order to avoid CPU cache issues but this may impact other programs running at the same time.

This system will implicitly look for the following env vars to configure the CPU usage limit:

(In priority order)

- `CFAVML_NUM_THREADS`
- `OMP_NUM_THREADS`  (Enable feature `env-var-compat`)
- `OPENBLAS_NUM_THREADS`  (Enable feature `env-var-compat`)

If no env var is provided, the system will use the number of **Physical** CPU cores available
as provided by the `num_cpus` crate. NOTE: This will always hard cap at the number of physical CPUs.

There is also some addition env vars used for configuration:

- `CFAVML_DEBUG` - If set to `true`/`1` this will log extra debug info, errors, etc... 
- `CFAVML_NO_CACHE_THREADPOOL` - If set to `true`/`1` the system will not cache created threadpools.
- `CFAVML_NO_PINNING` - If set to `true`/`1` the system will **not** pin created threads to cores.
  * This is **NOT** recommended as it severely impacts the performance of routines and general
    usage due to increased cache evictions/misses.