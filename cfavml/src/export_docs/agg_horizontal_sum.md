Performs a horizontal sum of all elements in vector `a` of size `dims` returning the total.

### Pseudocode

```ignore
result = 0

for i in range(dims):
    result += a[i]

return result
```

# Safety

This routine assumes: