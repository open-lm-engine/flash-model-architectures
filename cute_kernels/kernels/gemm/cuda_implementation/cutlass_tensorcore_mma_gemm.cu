#include <iostream>
#include <sstream>
#include <vector>

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/device_memory.h"
#include "include/dtypes/all.h"

using input_dtype = fp32;

template <bool is_A_transposed, bool is_B_transposed>
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

    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;

    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 16>;
    using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 16>;
    using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;

    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
    using EpilogueOp =
        cutlass::epilogue::thread::LinearCombination<accumulator_dtype,
                                                     128 / cutlass::sizeof_bits<accumulator_dtype>::value,
                                                     accumulator_dtype,
                                                     fp32>;

    constexpr int NumStages = 4;

    using CutlassGemm = cutlass::gemm::device::Gemm<input_dtype,
                                                    layout_A,
                                                    input_dtype,
                                                    layout_B,
                                                    input_dtype,
                                                    layout_C,
                                                    accumulator_dtype,
                                                    MMAOp,
                                                    SmArch,
                                                    ShapeMMAThreadBlock,
                                                    ShapeMMAWarp,
                                                    ShapeMMAOp,
                                                    EpilogueOp,
                                                    SwizzleThreadBlock,
                                                    NumStages>;

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
                                         {alpha, beta},
                                         {1});

    size_t workspace_size = CutlassGemm::get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_operator.can_implement(args);
    status = gemm_operator.initialize(args, workspace.get());

    gemm_operator();

    // call the kernel
    // cutlass::Status status = gemm_operator(args);
}

void cutlass_gemm_cuda(const torch::Tensor &A,
                       const torch::Tensor &B,
                       std::optional<torch::Tensor> &C,
                       torch::Tensor &output,
                       const bool &is_A_transposed,
                       const bool &is_B_transposed,
                       const fp32 &alpha,
                       const fp32 &beta,
                       const uint32 &M,
                       const uint32 &K,
                       const uint32 &N) {
    const input_dtype *A_data = reinterpret_cast<input_dtype *>(A.data_ptr<scalar_t>());
    const input_dtype *B_data = reinterpret_cast<input_dtype *>(B.data_ptr<scalar_t>());
    const input_dtype *C_data = C.has_value() ? reinterpret_cast<input_dtype *>(C.value().data_ptr<scalar_t>())
                                              : nullptr;
    input_dtype *output_data = reinterpret_cast<input_dtype *>(output.data_ptr<scalar_t>());

    const int32 _M = safe_cast_uint32_to_int32(M);
    const int32 _K = safe_cast_uint32_to_int32(K);
    const int32 _N = safe_cast_uint32_to_int32(N);

    if (is_A_transposed) {
        if (is_B_transposed) {
            _cutlass_gemm_templated_layout<true, true>(A_data, B_data, C_data, output_data, alpha, beta, _M, _K, _N);
        } else {
            _cutlass_gemm_templated_layout<true, false>(A_data, B_data, C_data, output_data, alpha, beta, _M, _K, _N);
        }
    } else {
        if (is_B_transposed) {
            _cutlass_gemm_templated_layout<false, true>(A_data, B_data, C_data, output_data, alpha, beta, _M, _K, _N);
        } else {
            _cutlass_gemm_templated_layout<false, false>(A_data, B_data, C_data, output_data, alpha, beta, _M, _K, _N);
        }
    }
}
