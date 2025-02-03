#include <iostream>
#include <sstream>
#include <vector>

#include "cutlass/gemm/device/gemm.h"

void gemm_fp32(int M, int K, int N, float alpha, float const *A, float const *B, float beta, float *C) {
    using RowMajor = cutlass::layout::RowMajor;
    using CutlassGemm = cutlass::gemm::device::Gemm<float, RowMajor, float, RowMajor, float, RowMajor>;

    CutlassGemm gemm_operator;
    CutlassGemm::Arguments args({M, N, K}, {A, M}, {B, K}, {C, M}, {C, M}, {alpha, beta});

    // call the kernel
    cutlass::Status status = gemm_operator(args);
}

void cutlass_gemm_cuda(const torch::Tensor &a,
                       const torch::Tensor &b,
                       std::optional<torch::Tensor> &c,
                       torch::Tensor &output,
                       const bool &is_a_transposed,
                       const bool &is_b_transposed,
                       const fp32 alpha,
                       const fp32 beta,
                       const uint32 &M,
                       const uint32 &K,
                       const uint32 &N,
                       const uint32 &BLOCK_SIZE) {
    dim3 NUM_BLOCKS = dim3(ceil_divide<uint32>(N, BLOCK_SIZE), ceil_divide<uint32>(M, BLOCK_SIZE), 1);
    dim3 BLOCK_SIZE_dim = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(a.scalar_type(), "gemm_fp32", ([&] { gemm_fp32(M, K, N, alpha, a, b, beta, c); }));
}
