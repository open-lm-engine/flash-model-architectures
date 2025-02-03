#include <iostream>
#include <sstream>
#include <vector>

#include "cutlass/gemm/device/gemm.h"
#include "dtypes/all.h"

template <typename scalar_t>
void _cutlass_gemm_cuda(const scalar_t *a,
                        const scalar_t *b,
                        const scalar_t *c,
                        scalar_t *output,
                        const fp32 alpha,
                        const fp32 beta,
                        const uint32 M,
                        const uint32 K,
                        const uint32 N) {
    using RowMajor = cutlass::layout::RowMajor;
    using CutlassGemm = cutlass::gemm::device::Gemm<fp32, RowMajor, fp32, RowMajor, fp32, RowMajor>;

    CutlassGemm gemm_operator;
    CutlassGemm::Arguments args({M, N, K}, {a, M}, {b, K}, {c, M}, {output, M}, {alpha, beta});

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
    TORCH_CHECK((BLOCK_SIZE * BLOCK_SIZE) % WARP_SIZE == 0);

    TORCH_CHECK(!is_a_transposed);
    TORCH_CHECK(!is_b_transposed);

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(
        a.scalar_type(), "gemm_fp32", ([&] { _cutlass_gemm_cuda<scalar_t>(a, b, c, output, alpha, beta, M, K, N); }));
}
