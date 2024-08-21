Checks each element pair from vectors `a` and `b` of size `dims` 
comparing if element `a` is **_equal to_** element `b` returning a mask vector of the same type.

### Pseudocode

```ignore
mask = [0; dims]

for i in range(dims):
    mask[i] = a[i] == b[i] ? 1 : 0

return mask
```

### Note on `NaN` handling on `f32/f64` types

For `f32` and `f64` types, `NaN` values are handled as always being `false` in **ANY** comparison.
Even when compared against each other.

- `0.0 == 0.0 -> true`
- `0.0 == NaN -> false`
- `NaN == NaN -> false`

# Safety

This routine assumes: