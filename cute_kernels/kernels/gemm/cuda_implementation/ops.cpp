#include <torch/extension.h>

void naive_gemm_cuda(const torch::Tensor &a,
                     const torch::Tensor &b,
                     torch::Tensor &c,
                     const bool &is_a_transposed,
                     const bool &is_b_transposed,
                     const uint32 &M,
                     const uint32 &K,
                     const uint32 &N,
                     const uint32 &BLOCK_SIZE_M,
                     const uint32 &BLOCK_SIZE_N);
