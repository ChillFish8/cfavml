import time
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np


N = 4096
x = np.random.randn(N, N).astype(np.float32)
y = np.random.randn(N, N).astype(np.float32)

flop = N*N*2*N
print(f"rust matrixmul: {(flop / 0.1971) * 1e-9} GFLOPS")

for _ in range(0, 10):
    start = time.perf_counter()
    res = np.matmul(x, y)
    elapsed = time.perf_counter() - start

    print(f"{(flop / elapsed) * 1e-9} GFLOPS {elapsed * 1e3 :.2f}ms")


sample_m1 = np.array([
    [1, 3, 2],
    [2, 0, 1],
    [1, 2, 0],
], dtype=np.int32)
sample_m2 = np.array([
    [3, 0, 1, 1],
    [1, 0, 1, 1],
    [0, 1, 2, 1],
], dtype=np.int32)

print(sample_m2.shape)
print(sample_m2[0, 1], sample_m2[1, 2])

print(sample_m1, sample_m2, sep="\n")
print(np.matmul(sample_m1, sample_m2, ))