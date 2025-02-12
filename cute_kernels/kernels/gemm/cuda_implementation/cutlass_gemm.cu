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
    using input_dtype = typename DType<scalar_t>::cutlass_dtype;
    using accumulator_dtype = fp32;

    using CutlassGemm = cutlass::gemm::device::
        Gemm<input_dtype, RowMajor, input_dtype, RowMajor, input_dtype, RowMajor, accumulator_dtype>;

    CutlassGemm gemm_operator;
    typename CutlassGemm::Arguments args({M, N, K},
                                         {reinterpret_cast<const input_dtype *>(a), K},
                                         {reinterpret_cast<const input_dtype *>(b), N},
                                         {reinterpret_cast<const input_dtype *>(c), N},
                                         {reinterpret_cast<input_dtype *>(output), N},
                                         {alpha, beta});

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

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(a.scalar_type(), "gemm_fp32", ([&] {
                                       _cutlass_gemm_cuda<scalar_t>(
                                           a.data_ptr<scalar_t>(),
                                           b.data_ptr<scalar_t>(),
                                           c.has_value() ? c.value().data_ptr<scalar_t>() : nullptr,
                                           output.data_ptr<scalar_t>(),
                                           alpha,
                                           beta,
                                           M,
                                           K,
                                           N);
                                   }));
}
