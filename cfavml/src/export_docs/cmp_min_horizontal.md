Finds the minimum element contained within vector `a` of size `dims` returning the result.

### Pseudocode

```ignore
result = inf

for i in range(dims):
    result = min(result, a[i])

return result
```

# Safety

This routine assumes: