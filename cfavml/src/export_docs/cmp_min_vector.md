Takes the element wise min of vectors `a` and `b` of size `dims` and stores the result
in `result` of size `dims`.

### Pseudocode

```ignore
result = [0; dims]

for i in range(dims):
    result[i] = min(a[i], b[i])

return result
```

# Safety

This routine assumes: