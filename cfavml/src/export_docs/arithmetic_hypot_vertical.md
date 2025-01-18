Performs an elementwise hypotenuse of input buffers `a` and `b` that can
be projected to the desired output size of `result`. Implementation is an appoximation that
_should_ match std::hypot in most cases. However, with some inputs it's been confirmed to be off by 1 ulp. Note: for `no_std` builds the result will be off more significantly for fallback

### Projecting Vectors

CFAVML allows for working over a wide variety of buffers for applications, projection is effectively
broadcasting of two input buffers implementing `IntoMemLoader<T>`.

By default, you can provide _two slices_, _one slice and a broadcast value_, or _two broadcast values_,
which exhibit the standard behaviour as you might expect.

When providing two slices as inputs they cannot be projected to a buffer
that is larger their input sizes by default. This means providing two slices
of `128` elements in length must take a result buffer of `128` elements in length.

### Implementation Pseudocode

_This is the (simplified) logic of the routine being called._

```ignore
result = [0; dims]

// assume |a|<|b| for all a,b
for i in range(dims):
    let hi = a[i].abs()
    let lo = b[i].abs()
    let scale = 1/hi
    result[i] = hi * sqrt(lo*scale + 1)

return result
```

# Panics

If vectors `a` and `b` cannot be projected to the target size of `result`.
Note that the projection rules are tied to the `MemLoader` implementation.

# Safety

This routine assumes:
