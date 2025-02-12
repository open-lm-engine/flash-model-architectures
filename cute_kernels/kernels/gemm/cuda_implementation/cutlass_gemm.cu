#include <iostream>
#include <sstream>
#include <vector>

#include "cutlass/gemm/device/gemm.h"
#include "include/dtypes/all.h"

template <typename scalar_t>
inline void _cutlass_gemm_cuda(const scalar_t *a,
                               const scalar_t *b,
                               const scalar_t *c,
                               scalar_t *output,
                               const fp32 &alpha,
                               const fp32 &beta,
                               const int &M,
                               const int &K,
                               const int &N) {
    // PyTorch tensors are row major
    using RowMajor = cutlass::layout::RowMajor;

    using element_A = fp32;
    using element_B = fp32;
    using element_C = fp32;

    using CutlassGemm = cutlass::gemm::device::Gemm<element_A, RowMajor, element_B, RowMajor, element_C, RowMajor>;

    CutlassGemm gemm_operator;
    CutlassGemm::Arguments args({M, N, K}, {a, K}, {b, N}, {c, N}, {output, N}, {alpha, beta});

    // call the kernel
    cutlass::Status status = gemm_operator(args);
}

void cutlass_gemm_cuda(const torch::Tensor &a,
                       const torch::Tensor &b,
                       std::optional<torch::Tensor> &c,
                       torch::Tensor &output,
                       const bool &is_a_transposed,
                       const bool &is_b_transposed,
                       const fp32 &alpha,
                       const fp32 &beta,
                       const uint32 &M,
                       const uint32 &K,
                       const uint32 &N) {
    TORCH_CHECK(!is_a_transposed);
    TORCH_CHECK(!is_b_transposed);
    _cutlass_gemm_cuda<fp32>(a.data_ptr<fp32>(),
                             b.data_ptr<fp32>(),
                             c.has_value() ? c.value().data_ptr<fp32>() : nullptr,
                             output.data_ptr<fp32>(),
                             alpha,
                             beta,
                             M,
                             K,
                             N);
    // AT_DISPATCH_CUSTOM_FLOAT_TYPES(
    //     fp32, "gemm_fp32", ([&] { _cutlass_gemm_cuda<scalar_t>(a.data_ptr<fp32>(), b.data_ptr<fp32>(),
    //     c.data_ptr<fp32>(), output.data_ptr<fp32>(), alpha, beta, M, K, N); }));
}
