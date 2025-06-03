// **************************************************
// Copyright (c) 2025, Mayank Mishra
// **************************************************

#include <torch/extension.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/tensor_view_io.h"
#include "include/cute_kernels.h"

namespace ck = cute_kernels;
using namespace cute;

using uint32 = ck::uint32;
using uint64 = ck::uint64;
using int32 = ck::int32;
using int64 = ck::int64;
using fp32 = ck::fp32;
using bf16 = ck::bf16;
using fp64 = ck::fp64;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int32, int32, int32>>;  // <M,N,K> per group
using ColumnMajor = cutlass::layout::ColumnMajor;
using RowMajor = cutlass::layout::RowMajor;

#define MAX_NUM_GROUPS 1024

// TODO modify this kernel to use vector load/stores
template <typename StrideA,
          typename StrideB,
          typename StrideC,
          bool is_A_transposed,
          bool is_B_transposed,
          typename UnderlyingProblemShape>
__global__ void populate_strides_cuda_kernel(const uint32 *M_array,
                                             const uint32 *N_array,
                                             const uint32 *K_array,
                                             StrideA *stride_A,
                                             StrideB *stride_B,
                                             StrideC *stride_C,
                                             int64 *offsets,
                                             UnderlyingProblemShape *problem_sizes,
                                             const uint32 E) {
    uint32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    int64 *offsets_A = offsets;
    int64 *offsets_B = offsets_A + E + 1;
    int64 *offsets_C = offsets_B + E + 1;

    if (thread_id == 0) {
        offsets_A[0] = 0;
        offsets_B[0] = 0;
        offsets_C[0] = 0;
    }

    if (thread_id < E) {
        const uint32 M = M_array[thread_id];
        const uint32 N = N_array[thread_id];
        const uint32 K = K_array[thread_id];

        stride_A[thread_id] = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
        stride_B[thread_id] = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
        stride_C[thread_id] = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
        problem_sizes[thread_id] = {M, N, K};

        thread_id++;
        offsets_A[thread_id] = M * K;
        offsets_B[thread_id] = K * N;
        offsets_C[thread_id] = M * N;
    }
}

template <typename ElementA, typename ElementB, typename ElementC, typename ElementD>
__global__ void offset_pointers_cuda_kernel(const ElementA **ptr_A,
                                            const ElementB **ptr_B,
                                            const ElementC **ptr_C,
                                            ElementD **ptr_D,
                                            const ElementA *A,
                                            const ElementB *B,
                                            const ElementC *C,
                                            ElementD *D,
                                            const int64 *offsets,
                                            const uint32 E) {
    const uint32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    const int64 *offsets_A = offsets;
    const int64 *offsets_B = offsets_A + E + 1;
    const int64 *offsets_C = offsets_B + E + 1;

    if (thread_id < E) {
        ptr_A[thread_id] = A + offsets_A[thread_id];
        ptr_B[thread_id] = B + offsets_B[thread_id];
        ptr_C[thread_id] = C + offsets_C[thread_id];
        ptr_D[thread_id] = D + offsets_C[thread_id];
    }
}

template <bool is_A_transposed, bool is_B_transposed>
inline void _grouped_gemm_cuda(const torch::Tensor &_A,
                               const torch::Tensor &_B,
                               const std::optional<torch::Tensor> &_C,
                               torch::Tensor &_D,
                               const torch::Tensor &M_array,
                               const torch::Tensor &N_array,
                               const torch::Tensor &K_array,
                               const fp32 &alpha,
                               const fp32 &beta,
                               const uint32 &E) {
    using ElementA = cutlass::bfloat16_t;
    using ElementB = cutlass::bfloat16_t;
    using ElementC = cutlass::bfloat16_t;
    using ElementD = cutlass::bfloat16_t;
    using ElementAccumulator = fp32;

    constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

    using LayoutA = std::conditional_t<is_A_transposed, ColumnMajor, RowMajor>;
    using LayoutB = std::conditional_t<is_B_transposed, ColumnMajor, RowMajor>;
    using LayoutC = RowMajor;

    using ArchTag = cutlass::arch::Sm100;
    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using StageCountType = cutlass::gemm::collective::StageCountAuto;

    using ClusterShape = Shape<int32, int32, _1>;
    using MmaTileShape = Shape<_256, _256, Int<128 / sizeof(ElementA)>>;

    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag,
        OperatorClass,
        MmaTileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementAccumulator,
        ElementC,
        LayoutC *,
        AlignmentC,
        ElementC,
        LayoutC *,
        AlignmentC,
        EpilogueSchedule,
        cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAccumulator>>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag,
        OperatorClass,
        ElementA,
        LayoutA *,
        AlignmentA,
        ElementB,
        LayoutB *,
        AlignmentB,
        ElementAccumulator,
        MmaTileShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    // Reference device GEMM implementation type
    using DeviceGemmReference = cutlass::reference::device::
        Gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator, ElementAccumulator>;

    using StrideA = typename Gemm::GemmKernel::InternalStrideA;
    using StrideB = typename Gemm::GemmKernel::InternalStrideB;
    using StrideC = typename Gemm::GemmKernel::InternalStrideC;

    // Device-side allocations
    cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;

    cutlass::DeviceAllocation<const typename Gemm::ElementA *> ptr_A;
    cutlass::DeviceAllocation<const typename Gemm::ElementB *> ptr_B;
    cutlass::DeviceAllocation<const typename Gemm::ElementC *> ptr_C;
    cutlass::DeviceAllocation<typename Gemm::ElementC *> ptr_D;

    cutlass::DeviceAllocation<StrideA> stride_A;
    cutlass::DeviceAllocation<StrideB> stride_B;
    cutlass::DeviceAllocation<StrideC> stride_C;

    using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100GroupParams<
        typename ProblemShape::UnderlyingProblemShape>::RasterOrderOptions;

    dim3 cluster_shape = dim3(4, 2, 1);
    dim3 cluster_shape_fallback = dim3(2, 1, 1);

    RasterOrderOptions raster_order = RasterOrderOptions::AlongM;

    ElementA *A;
    ElementB *B;
    ElementC *C;
    ElementC *D;

    DISPATCH_FLOAT_KERNEL(_A.scalar_type(), "get_raw_pointers", scalar_t, ([&] {
                              A = reinterpret_cast<ElementA *>(_A.data_ptr<scalar_t>());
                              B = reinterpret_cast<ElementB *>(_B.data_ptr<scalar_t>());
                              C = _C.has_value() ? reinterpret_cast<ElementC *>(_C.value().data_ptr<scalar_t>())
                                                 : nullptr;
                              D = reinterpret_cast<ElementC *>(_D.data_ptr<scalar_t>());
                          }));

    problem_sizes.reset(E);
    stride_A.reset(E);
    stride_B.reset(E);
    stride_C.reset(E);
    ptr_A.reset(E);
    ptr_B.reset(E);
    ptr_C.reset(E);
    ptr_D.reset(E);

    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kLong).device(_A.device());
    torch::Tensor offsets = torch::empty({3, E + 1}, options);

    DISPATCH_FLOAT_KERNEL(_A.scalar_type(), "populate_strides_cuda", scalar_t, ([&] {
                              populate_strides_cuda_kernel<StrideA,
                                                           StrideB,
                                                           StrideC,
                                                           is_A_transposed,
                                                           is_B_transposed,
                                                           typename ProblemShape::UnderlyingProblemShape>
                                  <<<1, 1024>>>(M_array.data_ptr<uint32>(),
                                                N_array.data_ptr<uint32>(),
                                                K_array.data_ptr<uint32>(),
                                                stride_A.get(),
                                                stride_B.get(),
                                                stride_C.get(),
                                                offsets.data_ptr<int64>(),
                                                problem_sizes.get(),
                                                E);
                          }));

    offsets = torch::cumsum(offsets, -1);

    DISPATCH_FLOAT_KERNEL(
        _A.scalar_type(), "offset_pointers_cuda", scalar_t, ([&] {
            offset_pointers_cuda_kernel<ElementA, ElementB, ElementC, ElementD><<<1, 1024>>>(
                ptr_A.get(), ptr_B.get(), ptr_C.get(), ptr_D.get(), A, B, C, D, offsets.data_ptr<int64>(), E);
        }));

    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm;

    cutlass::KernelHardwareInfo hw_info;
    cudaGetDevice(&hw_info.device_id);
    cudaDeviceGetAttribute(&hw_info.sm_count, cudaDevAttrMultiProcessorCount, hw_info.device_id);

    if (!is_static_v<ClusterShape>) {
        if (size<0>(typename Gemm::GemmKernel::CollectiveMainloop::AtomThrShapeMNK{}) == 2 &&
            (cluster_shape.x < 2 || cluster_shape_fallback.x < 2)) {
            std::cout << "Error: MMA2SMConfig kernel config needs cluster_dim.x >= 2" << std::endl;
        }
        hw_info.cluster_shape = cluster_shape;
        hw_info.cluster_shape_fallback = cluster_shape_fallback;
    }

    typename Gemm::Arguments arguments;
    decltype(arguments.epilogue.thread) fusion_args;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;

    // Single alpha / beta for all groups
    fusion_args.alpha = alpha;
    fusion_args.beta = beta;
    fusion_args.alpha_ptr_array = nullptr;
    fusion_args.beta_ptr_array = nullptr;
    fusion_args.dAlpha = {_0{}, _0{}, 0};
    fusion_args.dBeta = {_0{}, _0{}, 0};

    typename Gemm::GemmKernel::TileSchedulerArguments scheduler;
    scheduler.raster_order = raster_order;

    arguments = typename Gemm::Arguments{cutlass::gemm::GemmUniversalMode::kGrouped,
                                         {static_cast<int>(E), problem_sizes.get(), nullptr},
                                         {ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get()},
                                         {fusion_args, ptr_C.get(), stride_C.get(), ptr_D.get(), stride_C.get()},
                                         hw_info,
                                         scheduler};

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Check if the problem size is supported or not
    gemm.can_implement(arguments);

    // Initialize CUTLASS kernel with arguments and workspace pointer
    gemm.initialize(arguments, workspace.get());

    // Correctness / Warmup iteration
    gemm.run(/* stream = */ nullptr, /* cuda_adapter = */ nullptr, /* launch_with_pdl = */ false);
}

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
    // the addition of C is incorrent right now so just raise an error
    TORCH_CHECK(beta == 0);
    TORCH_CHECK(!_C.has_value());

    const uint32 E = M_array.numel();
    TORCH_CHECK(E <= MAX_NUM_GROUPS)
    TORCH_CHECK(N_array.numel() == E);
    TORCH_CHECK(K_array.numel() == E);

    if (is_A_transposed) {
        if (is_B_transposed) {
            _grouped_gemm_cuda<true, true>(_A, _B, _C, _D, M_array, N_array, K_array, alpha, beta, E);
        } else {
            _grouped_gemm_cuda<true, false>(_A, _B, _C, _D, M_array, N_array, K_array, alpha, beta, E);
        }
    } else {
        if (is_B_transposed) {
            _grouped_gemm_cuda<false, true>(_A, _B, _C, _D, M_array, N_array, K_array, alpha, beta, E);
        } else {
            _grouped_gemm_cuda<false, false>(_A, _B, _C, _D, M_array, N_array, K_array, alpha, beta, E);
        }
    }
}
