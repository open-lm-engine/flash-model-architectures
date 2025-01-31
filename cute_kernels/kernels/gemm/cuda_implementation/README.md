# GEMM CUDA

| Algorithm    | GMEM accesses | SMEM accesses |
| -------- | ------- | - |
| Naive | $MN(2K+1)$ | 0 |
| Shared Memory | $MN(\frac{2K}{B} + 1)$ | $2KB^2$ |
