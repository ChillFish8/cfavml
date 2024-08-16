Subtracts a single value from each element in vector `a` of size `dims`.

### Pseudocode

```ignore
result = [0; dims]

for i in range(dims):
    result[i] = a[i] - value

return result
```

# Safety

This routine assumes: