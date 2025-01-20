#include <torch/extension.h>

void naive_gemm_cuda(const torch::Tensor &a,
                     const torch::Tensor &b,
                     torch::Tensor &c,
                     const uint32 &BLOCK_SIZE_M,
                     const uint32 &BLOCK_SIZE_N);
