#include <iostream>
#include <sstream>
#include <vector>

#include "cutlass/gemm/device/gemm.h"
#include "include/dtypes/all.h"

template <typename input_dtype, bool is_a_transposed, bool is_b_transposed>
inline void _cutlass_gemm_templated_layout(const input_dtype *a,
                                           const input_dtype *b,
                                           const input_dtype *c,
                                           input_dtype *output,
                                           const fp32 &alpha,
                                           const fp32 &beta,
                                           const int32 &M,
                                           const int32 &K,
                                           const int32 &N) {
    // PyTorch tensors are row major
    using RowMajor = cutlass::layout::RowMajor;
    using ColumnMajor = cutlass::layout::ColumnMajor;

    using layout_a = std::conditional_t<is_a_transposed, ColumnMajor, RowMajor>;
    using layout_b = std::conditional_t<is_b_transposed, ColumnMajor, RowMajor>;
    using layout_c = RowMajor;

    using accumulator_dtype = fp32;

    using CutlassGemm = cutlass::gemm::device::
        Gemm<input_dtype, layout_a, input_dtype, layout_b, input_dtype, layout_c, accumulator_dtype>;

    const int32 leading_dimension_a = is_a_transposed ? M : K;
    const int32 leading_dimension_b = is_b_transposed ? K : N;
    const int32 leading_dimension_c = N;
    const int32 leading_dimension_output = N;

    CutlassGemm gemm_operator;
    typename CutlassGemm::Arguments args({M, N, K},
                                         {a, leading_dimension_a},
                                         {b, leading_dimension_b},
                                         {c, leading_dimension_c},
                                         {output, leading_dimension_output},
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
    AT_DISPATCH_CUSTOM_FLOAT_TYPES(
        a.scalar_type(), "cutlass_gemm_cuda", ([&] {
            using input_dtype = typename DType<scalar_t>::cutlass_dtype;

            const input_dtype *a_data = reinterpret_cast<input_dtype *>(a.data_ptr<scalar_t>());
            const input_dtype *b_data = reinterpret_cast<input_dtype *>(b.data_ptr<scalar_t>());
            const input_dtype *c_data = c.has_value() ? reinterpret_cast<input_dtype *>(c.value().data_ptr<scalar_t>())
                                                      : nullptr;
            input_dtype *output_data = reinterpret_cast<input_dtype *>(output.data_ptr<scalar_t>());

            const int32 _M = M;
            const int32 _K = K;
            const int32 _N = N;

            if (is_a_transposed) {
                if (is_b_transposed) {
                    _cutlass_gemm_templated_layout<input_dtype, true, true>(
                        a_data, b_data, c_data, output_data, alpha, beta, _M, _K, _N);
                } else {
                    _cutlass_gemm_templated_layout<input_dtype, true, false>(
                        a_data, b_data, c_data, output_data, alpha, beta, _M, _K, _N);
                }
            } else {
                if (is_b_transposed) {
                    _cutlass_gemm_templated_layout<input_dtype, false, true>(
                        a_data, b_data, c_data, output_data, alpha, beta, _M, _K, _N);
                } else {
                    _cutlass_gemm_templated_layout<input_dtype, false, false>(
                        a_data, b_data, c_data, output_data, alpha, beta, _M, _K, _N);
                }
            }
        }));
}
