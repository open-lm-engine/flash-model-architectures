// **************************************************
// Copyright (c) 2025, Mayank Mishra
// **************************************************

#include <torch/extension.h>

void grouped_gemm_cuda(const torch::Tensor &A,
                       const torch::Tensor &B,
                       torch::Tensor &output,
                       const torch::Tensor &expert_offsets,
                       const float &alpha,
                       const float &beta);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("grouped_gemm_cuda", &grouped_gemm_cuda, "grouped GEMM (CUDA)"); }
