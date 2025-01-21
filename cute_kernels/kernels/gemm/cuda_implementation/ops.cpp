#include <torch/extension.h>

void naive_gemm_cuda(const torch::Tensor &a,
                     const torch::Tensor &b,
                     torch::Tensor &c,
                     const bool &is_a_transposed,
                     const bool &is_b_transposed,
                     const uint &M,
                     const uint &K,
                     const uint &N,
                     const uint &BLOCK_SIZE_M,
                     const uint &BLOCK_SIZE_N);
