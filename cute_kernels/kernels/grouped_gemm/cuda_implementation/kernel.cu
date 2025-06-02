// **************************************************
// Copyright (c) 2025, Mayank Mishra
// **************************************************

#include <torch/extension.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "include/cute_kernels.h"

namespace ck = cute_kernels;

using uint32 = ck::uint32;
using uint64 = ck::uint64;
using int32 = ck::int32;
using int64 = ck::int64;
using fp32 = ck::fp32;
using bf16 = ck::bf16;
using fp64 = ck::fp64;

#define MAX_NUM_GROUPS 1024

struct GpuTimer {
    cudaStream_t _stream_id;
    cudaEvent_t _start;
    cudaEvent_t _stop;

    /// Constructor
    GpuTimer() : _stream_id(0) {
        cudaEventCreate(&_start);
        cudaEventCreate(&_stop);
    }

    /// Destructor
    ~GpuTimer() {
        cudaEventDestroy(_start);
        cudaEventDestroy(_stop);
    }

    /// Start the timer for a given stream (defaults to the default stream)
    void start(cudaStream_t stream_id = 0) {
        _stream_id = stream_id;
        cudaEventRecord(_start, _stream_id);
    }

    /// Stop the timer
    void stop() { cudaEventRecord(_stop, _stream_id); }

    /// Return the elapsed time (in milliseconds)
    fp32 elapsed_millis() {
        fp32 elapsed = 0.0;
        cudaEventSynchronize(_stop);
        cudaEventElapsedTime(&elapsed, _start, _stop);
        return elapsed;
    }
};

using namespace cute;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int32, int32, int32>>;  // <M,N,K> per group

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using ElementAccumulator = fp32;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::ColumnMajor;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using StageCountType = cutlass::gemm::collective::StageCountAuto;

using ClusterShape = Shape<int32_t, int32_t, _1>;

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

std::vector<StrideA> stride_A_host;
std::vector<StrideB> stride_B_host;
std::vector<StrideC> stride_C_host;

// Device-side allocations
cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;

cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_ref_D;

cutlass::DeviceAllocation<const typename Gemm::ElementA *> ptr_A;
cutlass::DeviceAllocation<const typename Gemm::ElementB *> ptr_B;
cutlass::DeviceAllocation<const typename Gemm::ElementC *> ptr_C;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput *> ptr_D;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput *> ptr_ref_D;

cutlass::DeviceAllocation<StrideA> stride_A;
cutlass::DeviceAllocation<StrideB> stride_B;
cutlass::DeviceAllocation<StrideC> stride_C;

using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100GroupParams<
    typename ProblemShape::UnderlyingProblemShape>::RasterOrderOptions;

/// Compute performance in GFLOP/s
fp64 get_gflops(const fp64 &runtime_s,
                std::vector<typename ProblemShape::UnderlyingProblemShape> &problem_sizes_host) {
    // Number of real-valued multiply-adds
    uint64 fmas = 0;

    for (auto const &problem : problem_sizes_host) {
        fmas += static_cast<uint64>(get<0>(problem)) * static_cast<uint64>(get<1>(problem)) *
                static_cast<uint64>(get<2>(problem));
    }
    // Two flops per multiply-add
    uint64 flop = uint64(2) * uint64(fmas);
    fp64 gflop = fp64(flop) / fp64(1.0e9);
    return gflop / runtime_s;
}

/// Helper to initialize a block of device data
template <class Element>
void initialize_block(cutlass::DeviceAllocation<Element> &block, uint64 seed = 2023) {
    cutlass::reference::device::BlockFillRandomUniform(
        block.get(), block.size(), seed, static_cast<Element>(8), static_cast<Element>(-8), 0);
}

// TODO modify this kernel to use vector load/stores
template <typename StrideA, typename StrideB, typename StrideC>
__global__ void populate_strides_cuda_kernel(const uint32 *M_array,
                                             const uint32 *N_array,
                                             const uint32 *K_array,
                                             StrideA *stride_A,
                                             StrideB *stride_B,
                                             StrideC *stride_C,
                                             int64_t *offset_A_device,
                                             int64_t *offset_B_device,
                                             int64_t *offset_C_device,
                                             const uint32 E) {
    uint32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id == 0) {
        offset_A_device[0] = 0;
        offset_B_device[0] = 0;
        offset_C_device[0] = 0;
    }

    if (thread_id < E) {
        const uint32 M = M_array[thread_id];
        const uint32 N = N_array[thread_id];
        const uint32 K = K_array[thread_id];

        stride_A[thread_id] = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
        stride_B[thread_id] = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
        stride_C[thread_id] = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});

        thread_id++;
        offset_A_device[thread_id] = M * K;
        offset_B_device[thread_id] = K * N;
        offset_C_device[thread_id] = M * N;
    }
}

template <typename ElementA, typename ElementB, typename ElementC, typename ElementD>
__global__ void offset_pointers_kernel(const ElementA **output_pointers_A,
                                       const ElementB **output_pointers_B,
                                       const ElementC **output_pointers_C,
                                       ElementD **output_pointers_D,
                                       const ElementA *A,
                                       const ElementB *B,
                                       const ElementC *C,
                                       const ElementD *D,
                                       const int64_t *offsets_A,
                                       const int64_t *offsets_B,
                                       const int64_t *offsets_C,
                                       const uint32 E) {
    const uint32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < E) {
        output_pointers_A[thread_id] = A + offsets_A[thread_id];
        output_pointers_B[thread_id] = B + offsets_B[thread_id];
        output_pointers_C[thread_id] = C + offsets_C[thread_id];
        output_pointers_D[thread_id] = D + offsets_C[thread_id];
    }
}

/// Allocates device-side data
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> allocate(
    const ElementA *A,
    const ElementB *B,
    const ElementC *C,
    const ElementC *D,
    const std::vector<typename ProblemShape::UnderlyingProblemShape> &problem_sizes_host,
    const torch::Tensor &M_array,
    const torch::Tensor &N_array,
    const torch::Tensor &K_array) {
    uint64 total_elements_C = 0;
    const uint32 E = problem_sizes_host.size();

    for (uint32 i = 0; i < E; i++) {
        auto problem = problem_sizes_host.at(i);
        auto M = get<0>(problem);
        auto N = get<1>(problem);
        auto K = get<2>(problem);

        total_elements_C += M * N;

        stride_A_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1}));
        stride_B_host.push_back(cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1}));
        stride_C_host.push_back(cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1}));
    }

    block_ref_D.reset(total_elements_C);

    stride_A.reset(E);
    stride_B.reset(E);
    stride_C.reset(E);
    ptr_A.reset(E);
    ptr_B.reset(E);
    ptr_C.reset(E);
    ptr_D.reset(E);

    torch::Tensor offset_A_device = torch::empty({E + 1}).to(torch::kLong);
    torch::Tensor offset_B_device = torch::empty({E + 1}).to(torch::kLong);
    torch::Tensor offset_C_device = torch::empty({E + 1}).to(torch::kLong);

    populate_strides_cuda_kernel<StrideA, StrideB, StrideC><<<1, 1024>>>(M_array.data_ptr<uint32>(),
                                                                         N_array.data_ptr<uint32>(),
                                                                         K_array.data_ptr<uint32>(),
                                                                         stride_A.get(),
                                                                         stride_B.get(),
                                                                         stride_C.get(),
                                                                         offset_A_device.data_ptr<int64_t>(),
                                                                         offset_B_device.data_ptr<int64_t>(),
                                                                         offset_C_device.data_ptr<int64_t>(),
                                                                         E);
    offset_A_device = torch::cumsum(offset_A_device, 0);
    offset_B_device = torch::cumsum(offset_B_device, 0);
    offset_C_device = torch::cumsum(offset_C_device, 0);

    problem_sizes.reset(E);
    problem_sizes.copy_from_host(problem_sizes_host.data());

    offset_pointers_kernel<ElementA, ElementB, ElementC, typename Gemm::EpilogueOutputOp::ElementOutput>
        <<<1, 1024>>>(ptr_A.get(),
                      ptr_B.get(),
                      ptr_C.get(),
                      ptr_D.get(),
                      A,
                      B,
                      C,
                      D,
                      offset_A_device.data_ptr<int64_t>(),
                      offset_B_device.data_ptr<int64_t>(),
                      offset_C_device.data_ptr<int64_t>(),
                      E);

    return std::make_tuple(offset_A_device, offset_B_device, offset_C_device);
}

typename Gemm::Arguments args_from_options(
    const uint32 &E,
    const dim3 &cluster_shape,
    const dim3 &cluster_shape_fallback,
    const fp32 &alpha,
    const fp32 &beta,
    const RasterOrderOptions &raster_order,
    const std::vector<typename ProblemShape::UnderlyingProblemShape> &problem_sizes_host,
    bool host_problem_shapes_available = true) {
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
                                         {static_cast<int>(E),
                                          problem_sizes.get(),
                                          host_problem_shapes_available ? problem_sizes_host.data() : nullptr},
                                         {ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get()},
                                         {fusion_args, ptr_C.get(), stride_C.get(), ptr_D.get(), stride_C.get()},
                                         hw_info,
                                         scheduler};

    return arguments;
}

bool verify(ElementA *A,
            ElementB *B,
            ElementC *C,
            const fp32 &alpha,
            const fp32 &beta,
            const std::vector<typename ProblemShape::UnderlyingProblemShape> &problem_sizes_host,
            torch::Tensor &offset_A_device,
            torch::Tensor &offset_B_device,
            torch::Tensor &offset_C_device) {
    const uint32 E = problem_sizes_host.size();
    bool passed = true;

    for (uint32 i = 0; i < E; ++i) {
        auto problem = problem_sizes_host.at(i);
        auto M = get<0>(problem);
        auto N = get<1>(problem);
        auto K = get<2>(problem);
        cutlass::TensorRef ref_A(A + offset_A_device[i].item<int64_t>(), Gemm::LayoutA::packed({M, K}));
        cutlass::TensorRef ref_B(B + offset_B_device[i].item<int64_t>(), Gemm::LayoutB::packed({K, N}));
        cutlass::TensorRef ref_C(C + offset_C_device[i].item<int64_t>(), Gemm::LayoutC::packed({M, N}));
        cutlass::TensorRef ref_D(block_ref_D.get() + offset_C_device[i].item<int64_t>(),
                                 Gemm::LayoutD::packed({M, N}));

        // Create instantiation for device reference gemm kernel
        DeviceGemmReference gemm_reference;

        // Launch device reference gemm kernel
        gemm_reference({M, N, K}, alpha, ref_A, ref_B, beta, ref_C, ref_D);

        // Wait for kernel to finish
        cudaDeviceSynchronize();

        // Check if output from CUTLASS kernel and reference kernel are equal or not
        passed &= cutlass::reference::device::BlockCompareEqual(
            block_ref_D.get() + offset_C_device[i].item<int64_t>(), D + offset_C_device[i].item<int64_t>(), M * N);
    }

    return passed;
}

inline uint32 get_size_at_index(const std::optional<torch::Tensor> &_offsets,
                                const std::optional<uint32> &_size,
                                const uint32 &index) {
    return _offsets.has_value() ? _offsets.value()[index].item<int64>() : _size.value();
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
    const uint32 E = M_array.numel();
    TORCH_CHECK(E <= MAX_NUM_GROUPS)
    TORCH_CHECK(N_array.numel() == E);
    TORCH_CHECK(K_array.numel() == E);

    dim3 cluster_shape = dim3(4, 2, 1);
    dim3 cluster_shape_fallback = dim3(2, 1, 1);
    RasterOrderOptions raster_order = RasterOrderOptions::AlongM;
    std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;

    problem_sizes_host.reserve(E);
    for (int i = 0; i < E; i++) {
        const uint32 M = M_array[i].item<int64>();
        const uint32 N = N_array[i].item<int64>();
        const uint32 K = K_array[i].item<int64>();

        problem_sizes_host.push_back({M, N, K});
    }

    ElementA *A;
    ElementB *B;
    ElementC *C;
    ElementC *D;

    DISPATCH_FLOAT_KERNEL(_A.scalar_type(), "copy", scalar_t, ([&] {
                              A = reinterpret_cast<ElementA *>(_A.data_ptr<scalar_t>());
                              B = reinterpret_cast<ElementB *>(_B.data_ptr<scalar_t>());
                              C = _C.has_value() ? reinterpret_cast<ElementC *>(_C.value().data_ptr<scalar_t>())
                                                 : nullptr;
                              D = reinterpret_cast<ElementC *>(_D.data_ptr<scalar_t>());
                          }));

    auto [offset_A_device, offset_B_device, offset_C_device] =
        allocate(A, B, C, D, problem_sizes_host, M_array, N_array, K_array);

    const bool host_problem_shapes_available = false;

    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm;

    // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
    Gemm::Arguments arguments = args_from_options(E,
                                                  cluster_shape,
                                                  cluster_shape_fallback,
                                                  alpha,
                                                  beta,
                                                  raster_order,
                                                  problem_sizes_host,
                                                  host_problem_shapes_available);

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

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    const bool passed =
        verify(A, B, C, alpha, beta, problem_sizes_host, offset_A_device, offset_B_device, offset_C_device);

    std::cout << "  Disposition: " << (passed ? "Passed" : "Failed") << std::endl;

    const uint32 iterations = 10;
    if (iterations > 0) {
        GpuTimer timer;
        timer.start();
        for (int iter = 0; iter < iterations; ++iter) {
            gemm.initialize(arguments, workspace.get());
            gemm.run(/* stream = */ nullptr, /* cuda_adapter = */ nullptr, /* launch_with_pdl = */ false);
        }
        timer.stop();

        // Compute average setup and runtime and GFLOPs.
        fp64 gflops = get_gflops(fp64(timer.elapsed_millis()) / fp64(iterations) / 1000.0, problem_sizes_host);
        std::cout << "  TFLOPS      : " << gflops / 1000.0 << std::endl;
    }
}
