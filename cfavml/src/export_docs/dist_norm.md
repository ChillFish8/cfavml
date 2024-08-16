Calculates the squared L2 norm of vector `a` of size `dims`.

### Pseudocode

```ignore
result = 0;

for i in range(dims):
    result += a[i] ** 2

return result
```

# Safety

This routine assumes: