Checks each element pair of elements from vectors `a` and `b` comparing if
element `a` is **_less than or equal to_** element `b`, storing the output as `1` (true)
or `0` (false) in `result`.

Vectors `a` and `b` can be projected to the new size of `result` if the mem loader allows.

### Projecting Vectors

CFAVML allows for working over a wide variety of buffers for applications, projection is effectively
broadcasting of two input buffers implementing `IntoMemLoader<T>`.

By default, you can provide _two slices_, _one slice and a broadcast value_, or _two broadcast values_,
which exhibit the standard behaviour as you might expect.

When providing two slices as inputs they cannot be projected to a buffer
that is larger their input sizes by default. This means providing two slices
of `128` elements in length must take a result buffer of `128` elements in length.

### Pseudocode

```ignore
mask = [0; dims]

for i in range(dims):
    mask[i] = a[i] <= b[i] ? 1 : 0

return mask
```

### Note on `NaN` handling on `f32/f64` types

For `f32` and `f64` types, `NaN` values are handled as always being `false` in **ANY** comparison. 
Even when compared against each other.

- `0.0 <= 1.0 -> true`
- `1.0 <= 1.0 -> true`
- `0.0 <= NaN -> false`
- `NaN <= 1.0 -> false`
- `NaN <= NaN -> false`

# Panics

If vectors `a` and `b` cannot be projected to the target size of `result`.
Note that the projection rules are tied to the `MemLoader` implementation.

# Safety

This routine assumes: