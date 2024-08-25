Performs an element wise addition of two input buffers `a` and `b` that can
be projected to the desired output size of `result`.

### Projecting Vectors

CFAVML allows for working over a wide variety of buffers for applications, projection is effectively 
broadcasting of two input buffers implementing `IntoMemLoader<T>`.

By default, you can provide _two slices_, _one slice and a broadcast value_, or _two broadcast values_, 
which exhibit the standard behaviour as you might expect.

When providing two slices as inputs they cannot be projected to a buffer
that is larger their input sizes by default. This means providing two slices
of `128` elements in length must take a result buffer of `128` elements in length.

### Implementation Pseudocode

_This is the logic of the routine being called._

```ignore
result = [0; dims]

for i in range(dims):
    result[i] = a[i] + b[i]

return result
```

# Panics

If vectors `a` and `b` cannot be projected to the target size of `result`.
Note that the projection rules are tied to the `MemLoader` implementation.

# Safety

This routine assumes: