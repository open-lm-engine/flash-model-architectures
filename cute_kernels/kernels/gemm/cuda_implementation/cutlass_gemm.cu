#include <iostream>
#include <sstream>
#include <vector>

#include "cutlass/gemm/device/gemm.h"
#include "include/cute_kernels.h"

namespace ck = cute_kernels;

using fp32 = ck::fp32;
using uint32 = ck::uint32;
using int32 = ck::int32;

template <typename input_dtype, bool is_A_transposed, bool is_B_transposed>
inline void _cutlass_gemm_templated_layout(const input_dtype *A,
                                           const input_dtype *B,
                                           const input_dtype *C,
                                           input_dtype *output,
                                           const fp32 &alpha,
                                           const fp32 &beta,
                                           const int32 &M,
                                           const int32 &K,
                                           const int32 &N) {
    // PyTorch tensors are row major
    using RowMajor = cutlass::layout::RowMajor;
    using ColumnMajor = cutlass::layout::ColumnMajor;

    using layout_A = std::conditional_t<is_A_transposed, ColumnMajor, RowMajor>;
    using layout_B = std::conditional_t<is_B_transposed, ColumnMajor, RowMajor>;
    using layout_C = RowMajor;

    using accumulator_dtype = fp32;

    using CutlassGemm = cutlass::gemm::device::
        Gemm<input_dtype, layout_A, input_dtype, layout_B, input_dtype, layout_C, accumulator_dtype>;

    const int32 leading_dimension_A = is_A_transposed ? M : K;
    const int32 leading_dimension_B = is_B_transposed ? K : N;
    const int32 leading_dimension_C = N;
    const int32 leading_dimension_output = N;

    CutlassGemm gemm_operator;
    typename CutlassGemm::Arguments args({M, N, K},
                                         {A, leading_dimension_A},
                                         {B, leading_dimension_B},
                                         {C, leading_dimension_C},
                                         {output, leading_dimension_output},
                                         {alpha, beta});

    // call the kernel
    cutlass::Status status = gemm_operator(args);
}

void cutlass_gemm_cuda(const torch::Tensor &A,
                       const torch::Tensor &B,
                       std::optional<torch::Tensor> &_C,
                       torch::Tensor &output,
                       const bool &is_A_transposed,
                       const bool &is_B_transposed,
                       const fp32 &alpha,
                       const fp32 &beta,
                       const uint32 &M,
                       const uint32 &K,
                       const uint32 &N) {
    CHECK_CUDA_TENSOR(A);
    CHECK_CUDA_TENSOR(B);
    if (_C.has_value()) {
        CHECK_CUDA_TENSOR(_C.value());
    }
    CHECK_CUDA_TENSOR(output);

    DISPATCH_FLOAT_KERNEL(A.scalar_type(), "cutlass_gemm_cuda", scalar_t, ([&] {
                              using input_dtype = typename ck::DType<scalar_t>::cutlass_dtype;

                              const input_dtype *A_data = reinterpret_cast<input_dtype *>(A.data_ptr<scalar_t>());
                              const input_dtype *B_data = reinterpret_cast<input_dtype *>(B.data_ptr<scalar_t>());
                              const input_dtype *C_data =
                                  C.has_value() ? reinterpret_cast<input_dtype *>(_C.value().data_ptr<scalar_t>())
                                                : nullptr;
                              input_dtype *output_data = reinterpret_cast<input_dtype *>(output.data_ptr<scalar_t>());

                              const int32 _M = ck::safe_cast_uint32_to_int32(M);
                              const int32 _K = ck::safe_cast_uint32_to_int32(K);
                              const int32 _N = ck::safe_cast_uint32_to_int32(N);

                              if (is_A_transposed) {
                                  if (is_B_transposed) {
                                      _cutlass_gemm_templated_layout<input_dtype, true, true>(
                                          A_data, B_data, C_data, output_data, alpha, beta, _M, _K, _N);
                                  } else {
                                      _cutlass_gemm_templated_layout<input_dtype, true, false>(
                                          A_data, B_data, C_data, output_data, alpha, beta, _M, _K, _N);
                                  }
                              } else {
                                  if (is_B_transposed) {
                                      _cutlass_gemm_templated_layout<input_dtype, false, true>(
                                          A_data, B_data, C_data, output_data, alpha, beta, _M, _K, _N);
                                  } else {
                                      _cutlass_gemm_templated_layout<input_dtype, false, false>(
                                          A_data, B_data, C_data, output_data, alpha, beta, _M, _K, _N);
                                  }
                              }
                          }));
}
