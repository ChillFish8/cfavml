import os
import time

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np


N = 4096
x = np.random.randn(N, N).astype(np.float32)


for _ in range(0, 10):
    start = time.perf_counter()
    res = np.transpose(x).tobytes(order="C")
    elapsed = time.perf_counter() - start

    print(f"{(N*N / elapsed) * 1e-9} GFLOPS {elapsed * 1e3 :.2f}ms")