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
using uint8 = ck::uint8;
using fp32 = ck::fp32;
using bf16 = ck::bf16;
using fp64 = ck::fp64;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int32, int32, int32>>;  // <M,N,K> per group
using ColumnMajor = cutlass::layout::ColumnMajor;
using RowMajor = cutlass::layout::RowMajor;

#define MAX_NUM_GROUPS 1024

#define CUTLASS_CHECK(status)                                                                                        \
    {                                                                                                                \
        cutlass::Status error = status;                                                                              \
        if (error != cutlass::Status::kSuccess) {                                                                    \
            std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                                                                      \
        }                                                                                                            \
    }

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

template <typename ElementA, typename ElementB, typename ElementC, typename ElementD, bool has_C>
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
        if (has_C) {
            ptr_C[thread_id] = C + offsets_C[thread_id];
        }
        ptr_D[thread_id] = D + offsets_C[thread_id];
    }
}

/// Compute performance in GFLOP/s
fp64 get_gflops(const fp64 &runtime_s) {
    // Number of real-valued multiply-adds
    // Two flops per multiply-add
    uint64 flop = uint64(2) * uint64(4096) * uint64(4096) * uint64(512) * uint64(16);
    fp64 gflop = fp64(flop) / fp64(1.0e9);
    return gflop / runtime_s;
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
                               const uint32 &E,
                               const bool &benchmark) {
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

    torch::Tensor memory = torch::empty({6 * E + (_C.has_value() ? E : 0)},
                                        torch::TensorOptions().dtype(torch::kUInt64).device(_A.device()));

    StrideA *stride_A = reinterpret_cast<StrideA *>(memory.data_ptr<uint64>());
    StrideB *stride_B = reinterpret_cast<StrideB *>(memory.data_ptr<uint64>() + E);
    StrideC *stride_C = reinterpret_cast<StrideC *>(memory.data_ptr<uint64>() + 2 * E);

    const ElementA **ptr_A = reinterpret_cast<const ElementA **>(memory.data_ptr<uint64>() + 3 * E);
    const ElementB **ptr_B = reinterpret_cast<const ElementB **>(memory.data_ptr<uint64>() + 4 * E);
    ElementD **ptr_D = reinterpret_cast<ElementD **>(memory.data_ptr<uint64>() + 5 * E);
    const ElementC **ptr_C =
        reinterpret_cast<const ElementC **>(_C.has_value() ? memory.data_ptr<uint64>() + 6 * E : nullptr);

    torch::Tensor memory1 = torch::empty({3 * E}, torch::TensorOptions().dtype(torch::kUInt32).device(_A.device()));
    ProblemShape::UnderlyingProblemShape *problem_sizes =
        reinterpret_cast<ProblemShape::UnderlyingProblemShape *>(memory1.data_ptr<uint32>());

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
                                                stride_A,
                                                stride_B,
                                                stride_C,
                                                offsets.data_ptr<int64>(),
                                                problem_sizes,
                                                E);
                          }));

    offsets.cumsum_(-1);

    DISPATCH_FLOAT_KERNEL(
        _A.scalar_type(), "offset_pointers_cuda", scalar_t, ([&] {
            if (_C.has_value()) {
                offset_pointers_cuda_kernel<ElementA, ElementB, ElementC, ElementD, true>
                    <<<1, 1024>>>(ptr_A, ptr_B, ptr_C, ptr_D, A, B, C, D, offsets.data_ptr<int64>(), E);
            } else {
                offset_pointers_cuda_kernel<ElementA, ElementB, ElementC, ElementD, false>
                    <<<1, 1024>>>(ptr_A, ptr_B, ptr_C, ptr_D, A, B, C, D, offsets.data_ptr<int64>(), E);
            }
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
                                         {static_cast<int>(E), problem_sizes, nullptr},
                                         {ptr_A, stride_A, ptr_B, stride_B},
                                         {fusion_args, ptr_C, stride_C, ptr_D, stride_C},
                                         hw_info,
                                         scheduler};

    // allocate workspace memory
    const uint64 workspace_size = Gemm::get_workspace_size(arguments);
    torch::Tensor workspace =
        torch::empty({int64(workspace_size)}, torch::TensorOptions().dtype(at::kByte).device(_A.device()));

    // Check if the problem size is supported or not
    CUTLASS_CHECK(gemm.can_implement(arguments));

    // Initialize CUTLASS kernel with arguments and workspace pointer
    CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr()));

    // Correctness / Warmup iteration
    CUTLASS_CHECK(gemm.run(/* stream = */ nullptr, /* cuda_adapter = */ nullptr, /* launch_with_pdl = */ false));

    if (benchmark) {
        const uint32 iterations = 10;
        ck::GpuTimer timer;

        timer.start();

        for (int iter = 0; iter < iterations; ++iter) {
            gemm.initialize(arguments, workspace.data_ptr());
            gemm.run(/* stream = */ nullptr, /* cuda_adapter = */ nullptr, /* launch_with_pdl = */ false);
        }

        timer.stop();

        // Compute average setup and runtime and GFLOPs.
        std::cout << "  TFLOPS      : "
                  << get_gflops(fp64(timer.elapsed_millis()) / fp64(iterations) / 1000.0) / 1000.0 << std::endl;
    }
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
                       const fp32 &beta,
                       const bool &benchmark) {
    // the addition of C is incorrent right now so just raise an error
    TORCH_CHECK(beta == 0);
    TORCH_CHECK(!_C.has_value());

    const uint32 E = M_array.numel();
    TORCH_CHECK(E <= MAX_NUM_GROUPS)
    TORCH_CHECK(N_array.numel() == E);
    TORCH_CHECK(K_array.numel() == E);

    if (is_A_transposed) {
        if (is_B_transposed) {
            _grouped_gemm_cuda<true, true>(_A, _B, _C, _D, M_array, N_array, K_array, alpha, beta, E, benchmark);
        } else {
            _grouped_gemm_cuda<true, false>(_A, _B, _C, _D, M_array, N_array, K_array, alpha, beta, E, benchmark);
        }
    } else {
        if (is_B_transposed) {
            _grouped_gemm_cuda<false, true>(_A, _B, _C, _D, M_array, N_array, K_array, alpha, beta, E, benchmark);
        } else {
            _grouped_gemm_cuda<false, false>(_A, _B, _C, _D, M_array, N_array, K_array, alpha, beta, E, benchmark);
        }
    }
}
