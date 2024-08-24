Calculates the dot product between vectors `a` and `b`.

### Pseudocode

```ignore
result = 0;

for i in range(dims):
    result += a[i] * b[i]

return result
```

# Panics

If vectors `a` and `b` are not equal in the length.

# Safety

This routine assumes: