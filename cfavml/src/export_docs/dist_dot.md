Calculates the dot product between vectors `a` and `b` of size `dims`.

### Pseudocode

```ignore
result = 0;

for i in range(dims):
    result += a[i] * b[i]

return result
```

# Safety

This routine assumes: