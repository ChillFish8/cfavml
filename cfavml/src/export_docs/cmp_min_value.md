Takes the element wise min of vector `a` of size `dims` and the provided broadcast value, storing the result
in vector `result` of size `dims`.

### Pseudocode

```ignore
result = [0; dims]

for i in range(dims):
    result[i] = min(a[i], value)

return result
```

# Safety

This routine assumes: