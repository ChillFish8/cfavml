Calculates the cosine similarity distance between vectors `a` and `b` of size `dims`.

### Pseudocode

```ignore
result = 0
norm_a = 0
norm_b = 0

for i in range(dims):
    result += a[i] * b[i]
    norm_a += a[i] ** 2
    norm_b += b[i] ** 2

if norm_a == 0.0 and norm_b == 0.0:
    return 0.0
elif norm_a == 0.0 or norm_b == 0.0:
    return 1.0
else:
    return 1.0 - (result / sqrt(norm_a * norm_b))
```

# Safety

This routine assumes: