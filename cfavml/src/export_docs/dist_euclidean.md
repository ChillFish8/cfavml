Calculates the squared Euclidean distance between vectors `a` and `b` of size `dims`.

### Pseudocode

```ignore
result = 0;

for i in range(dims):
    diff = a[i] - b[i]
    result += diff ** 2

return result
```

# Safety

This routine assumes: