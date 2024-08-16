Performs an element wise addition of vectors `a` and `b` of size `dims` and store the result
in `result` vector of size `dims`.

### Pseudocode

```ignore
result = [0; dims]

for i in range(dims):
    result[i] = a[i] + b[i]

return result
```

# Safety

This routine assumes: