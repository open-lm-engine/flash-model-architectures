#include <torch/extension.h>

void naive_gemm_cuda(const torch::Tensor &A,
                     const torch::Tensor &B,
                     std::optional<torch::Tensor> &C,
                     torch::Tensor &output,
                     const bool &is_A_transposed,
                     const bool &is_B_transposed,
                     const float &alpha,
                     const float &beta,
                     const uint &M,
                     const uint &K,
                     const uint &N,
                     const uint &BLOCK_SIZE_M,
                     const uint &BLOCK_SIZE_N);

void shared_memory_gemm_cuda(const torch::Tensor &A,
                             const torch::Tensor &B,
                             std::optional<torch::Tensor> &C,
                             torch::Tensor &output,
                             const bool &is_A_transposed,
                             const bool &is_B_transposed,
                             const float &alpha,
                             const float &beta,
                             const uint &M,
                             const uint &K,
                             const uint &N,
                             const uint &BLOCK_SIZE);

void cutlass_gemm_cuda(const torch::Tensor &A,
                       const torch::Tensor &B,
                       std::optional<torch::Tensor> &C,
                       torch::Tensor &output,
                       const bool &is_A_transposed,
                       const bool &is_B_transposed,
                       const float &alpha,
                       const float &beta,
                       const uint &M,
                       const uint &K,
                       const uint &N);

void cutlass_tensorcore_mma_gemm_cuda(const torch::Tensor &A,
                                      const torch::Tensor &B,
                                      std::optional<torch::Tensor> &C,
                                      torch::Tensor &output,
                                      const bool &is_A_transposed,
                                      const bool &is_B_transposed,
                                      const float &alpha,
                                      const float &beta,
                                      const uint &M,
                                      const uint &K,
                                      const uint &N);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_gemm_cuda", &naive_gemm_cuda, "naive GEMM (CUDA)");
    m.def("shared_memory_gemm_cuda", &shared_memory_gemm_cuda, "shared memory GEMM (CUDA)");
    m.def("cutlass_gemm_cuda", &cutlass_gemm_cuda, "CUTLASS GEMM (CUDA)");
    m.def("cutlass_tensorcore_mma_gemm_cuda", &cutlass_tensorcore_mma_gemm_cuda, "CUTLASS tensorcore MMA GEMM (CUDA)");
}
