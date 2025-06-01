// **************************************************
// Copyright (c) 2025, Mayank Mishra
// **************************************************

#include <torch/extension.h>

void grouped_gemm_cuda(const torch::Tensor &A,
                       const torch::Tensor &B,
                       torch::Tensor &output,
                       const torch::Tensor &M_array,
                       const torch::Tensor &N_array,
                       const torch::Tensor &K_array,
                       const bool &is_A_transposed,
                       const bool &is_B_transposed,
                       const float &alpha,
                       const float &beta);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("grouped_gemm_cuda", &grouped_gemm_cuda, "grouped GEMM (CUDA)"); }
