Calculates the squared Euclidean distance between vectors `a` and `b`.

### Implementation Pseudocode

_This is the logic of the routine being called._

```ignore
result = 0;

for i in range(dims):
    diff = a[i] - b[i]
    result += diff ** 2

return result
```

# Panics

If vectors `a` and `b` are not equal in the length.

# Safety

This routine assumes: