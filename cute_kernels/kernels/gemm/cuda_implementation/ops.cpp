#include <torch/extension.h>

#include "../../../include/dtypes/alias.h"

void naive_gemm_cuda(const torch::Tensor &a,
                     const torch::Tensor &b,
                     torch::Tensor &c,
                     const bool &is_a_transposed,
                     const bool &is_b_transposed,
                     const fp32 alpha,
                     const fp32 beta,
                     const uint32 &M,
                     const uint32 &K,
                     const uint32 &N,
                     const uint32 &BLOCK_SIZE_M,
                     const uint32 &BLOCK_SIZE_N);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("naive_gemm_cuda", &naive_gemm_cuda, "naive GEMM (CUDA)"); }
