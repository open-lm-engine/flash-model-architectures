// **************************************************
// Copyright (c) 2025, Mayank Mishra
// **************************************************

#include <torch/extension.h>

void grouped_gemm_cuda(const torch::Tensor &_A,
                       const torch::Tensor &_B,
                       const std::optional<torch::Tensor> &_C,
                       torch::Tensor &_D,
                       const torch::Tensor &M_array,
                       const torch::Tensor &N_array,
                       const torch::Tensor &K_array,
                       torch::Tensor &_ptr_A,
                       torch::Tensor &_ptr_B,
                       std::optional<torch::Tensor> &_ptr_C,
                       torch::Tensor &_ptr_D,
                       torch::Tensor &_problem_sizes,
                       const bool &is_A_transposed,
                       const bool &is_B_transposed,
                       const float &alpha,
                       const float &beta,
                       const bool &benchmark);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("grouped_gemm_cuda", &grouped_gemm_cuda, "grouped GEMM (CUDA)"); }
