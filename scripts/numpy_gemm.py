import time
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np


N = 2048
x = np.random.randn(N, N).astype(np.float32)
y = np.random.randn(N, N).astype(np.float32)

flop = N*N*2*N
print(f"rust matrixmul: {(flop / 0.1971) * 1e-9} GFLOPS")

print(f"{N*N*2*N / 1e9:.2f} GFLOP")
for _ in range(0, 10):
    start = time.perf_counter()
    res = np.matmul(x, y)
    elapsed = time.perf_counter() - start

    print(f"{(flop / elapsed) * 1e-9} GFLOPS")

