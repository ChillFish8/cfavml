Multiplies each element in vector `a` of size `dims` by a single value.

### Pseudocode

```ignore
result = [0; dims]

for i in range(dims):
    result[i] = a[i] * value

return result
```

# Safety

This routine assumes: