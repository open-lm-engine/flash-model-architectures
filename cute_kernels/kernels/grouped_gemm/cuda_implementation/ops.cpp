// **************************************************
// Copyright (c) 2025, Mayank Mishra
// **************************************************

#include <torch/extension.h>

void _grouped_gemm_cuda(const torch::Tensor &_A,
                        const torch::Tensor &_B,
                        const std::optional<torch::Tensor> &_C,
                        torch::Tensor &_D,
                        const torch::Tensor &M_array,
                        const torch::Tensor &N_array,
                        const torch::Tensor &K_array,
                        const float &alpha,
                        const float &beta);

void grouped_gemm_cuda(const torch::Tensor &_A,
                       const torch::Tensor &_B,
                       const std::optional<torch::Tensor> &_C,
                       torch::Tensor &_D,
                       const torch::Tensor &M_array,
                       const torch::Tensor &N_array,
                       const torch::Tensor &K_array,
                       const bool &is_A_transposed,
                       const bool &is_B_transposed,
                       const fp32 &alpha,
                       const fp32 &beta) {
    if (is_A_transposed) {
        if (is_B_transposed) {
            _grouped_gemm_cuda<true, true>(_A, _B, _C, _D, M_array, N_array, K_array, alpha, beta);
        } else {
            _grouped_gemm_cuda<true, false>(_A, _B, _C, _D, M_array, N_array, K_array, alpha, beta);
        }
    } else {
        if (is_B_transposed) {
            _grouped_gemm_cuda<false, true>(_A, _B, _C, _D, M_array, N_array, K_array, alpha, beta);
        } else {
            _grouped_gemm_cuda<false, false>(_A, _B, _C, _D, M_array, N_array, K_array, alpha, beta);
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("grouped_gemm_cuda", &grouped_gemm_cuda, "grouped GEMM (CUDA)"); }
