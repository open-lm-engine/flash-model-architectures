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
using int64 = ck::int64;

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
    float elapsed_millis() {
        float elapsed = 0.0;
        cudaEventSynchronize(_stop);
        cudaEventElapsedTime(&elapsed, _start, _stop);
        return elapsed;
    }
};

using namespace cute;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M,N,K> per group

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using ElementAccumulator = float;

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
using StrideD = typename Gemm::GemmKernel::InternalStrideD;

// Host-side allocations
std::vector<int64_t> offset_A;
std::vector<int64_t> offset_B;
std::vector<int64_t> offset_C;

std::vector<StrideA> stride_A_host;
std::vector<StrideB> stride_B_host;
std::vector<StrideC> stride_C_host;

// Device-side allocations
cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;

cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
cutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D;
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
double get_gflops(const double &runtime_s,
                  std::vector<typename ProblemShape::UnderlyingProblemShape> &problem_sizes_host) {
    // Number of real-valued multiply-adds
    uint64_t fmas = 0;

    for (auto const &problem : problem_sizes_host) {
        fmas += static_cast<uint64_t>(get<0>(problem)) * static_cast<uint64_t>(get<1>(problem)) *
                static_cast<uint64_t>(get<2>(problem));
    }
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * uint64_t(fmas);
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
}

/// Helper to initialize a block of device data
template <class Element>
void initialize_block(cutlass::DeviceAllocation<Element> &block, uint64_t seed = 2023) {
    cutlass::reference::device::BlockFillRandomUniform(
        block.get(), block.size(), seed, static_cast<Element>(8), static_cast<Element>(-8), 0);
}

/// Allocates device-side data
void allocate(const std::vector<typename ProblemShape::UnderlyingProblemShape> &problem_sizes_host) {
    int64_t total_elements_A = 0;
    int64_t total_elements_B = 0;
    int64_t total_elements_C = 0;

    const uint32 E = problem_sizes_host.size();

    for (uint32 i = 0; i < E; i++) {
        auto problem = problem_sizes_host.at(i);
        auto M = get<0>(problem);
        auto N = get<1>(problem);
        auto K = get<2>(problem);

        offset_A.push_back(total_elements_A);
        offset_B.push_back(total_elements_B);
        offset_C.push_back(total_elements_C);

        int64_t elements_A = M * K;
        int64_t elements_B = K * N;
        int64_t elements_C = M * N;

        total_elements_A += elements_A;
        total_elements_B += elements_B;
        total_elements_C += elements_C;

        stride_A_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1}));
        stride_B_host.push_back(cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1}));
        stride_C_host.push_back(cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1}));
    }

    block_A.reset(total_elements_A);
    block_B.reset(total_elements_B);
    block_C.reset(total_elements_C);
    block_D.reset(total_elements_C);
    block_ref_D.reset(total_elements_C);
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const float &alpha,
                const float &beta,
                const std::vector<typename ProblemShape::UnderlyingProblemShape> &problem_sizes_host) {
    const uint32 E = problem_sizes_host.size();

    problem_sizes.reset(E);
    problem_sizes.copy_from_host(problem_sizes_host.data());

    std::vector<ElementA *> ptr_A_host(E);
    std::vector<ElementB *> ptr_B_host(E);
    std::vector<ElementC *> ptr_C_host(E);
    std::vector<ElementC *> ptr_D_host(E);

    for (uint32 i = 0; i < E; ++i) {
        ptr_A_host.at(i) = block_A.get() + offset_A.at(i);
        ptr_B_host.at(i) = block_B.get() + offset_B.at(i);
        ptr_C_host.at(i) = block_C.get() + offset_C.at(i);
        ptr_D_host.at(i) = block_D.get() + offset_C.at(i);
    }

    ptr_A.reset(E);
    ptr_A.copy_from_host(ptr_A_host.data());

    ptr_B.reset(E);
    ptr_B.copy_from_host(ptr_B_host.data());

    ptr_C.reset(E);
    ptr_C.copy_from_host(ptr_C_host.data());

    ptr_D.reset(E);
    ptr_D.copy_from_host(ptr_D_host.data());

    stride_A.reset(E);
    stride_A.copy_from_host(stride_A_host.data());

    stride_B.reset(E);
    stride_B.copy_from_host(stride_B_host.data());

    stride_C.reset(E);
    stride_C.copy_from_host(stride_C_host.data());

    initialize_block(block_A, 2023);
    initialize_block(block_B, 2022);
    initialize_block(block_C, 2021);
}

typename Gemm::Arguments args_from_options(
    const uint32 &E,
    const dim3 &cluster_shape,
    const dim3 &cluster_shape_fallback,
    const float &alpha,
    const float &beta,
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

bool verify(const float &alpha,
            const float &beta,
            const std::vector<typename ProblemShape::UnderlyingProblemShape> &problem_sizes_host) {
    const uint32 E = problem_sizes_host.size();

    bool passed = true;
    for (uint32 i = 0; i < E; ++i) {
        auto problem = problem_sizes_host.at(i);
        auto M = get<0>(problem);
        auto N = get<1>(problem);
        auto K = get<2>(problem);
        cutlass::TensorRef ref_A(block_A.get() + offset_A.at(i), Gemm::LayoutA::packed({M, K}));
        cutlass::TensorRef ref_B(block_B.get() + offset_B.at(i), Gemm::LayoutB::packed({K, N}));
        cutlass::TensorRef ref_C(block_C.get() + offset_C.at(i), Gemm::LayoutC::packed({M, N}));
        cutlass::TensorRef ref_D(block_ref_D.get() + offset_C.at(i), Gemm::LayoutD::packed({M, N}));

        // Create instantiation for device reference gemm kernel
        DeviceGemmReference gemm_reference;

        // Launch device reference gemm kernel
        gemm_reference({M, N, K}, alpha, ref_A, ref_B, beta, ref_C, ref_D);

        // Wait for kernel to finish
        cudaDeviceSynchronize();

        // Check if output from CUTLASS kernel and reference kernel are equal or not
        passed &= cutlass::reference::device::BlockCompareEqual(
            block_ref_D.get() + offset_C.at(i), block_D.get() + offset_C.at(i), M * N);
    }
    return passed;
}

void grouped_gemm_cuda(const torch::Tensor &A,
                       const torch::Tensor &B,
                       torch::Tensor &output,
                       const torch::Tensor &expert_offsets,
                       const float &alpha,
                       const float &beta) {
    const uint32 E = B.size(0);
    const uint32 N = B.size(1);
    const uint32 K = B.size(2);
    const uint32 TK = A.size(0);

    dim3 cluster_shape = dim3(4, 2, 1);
    dim3 cluster_shape_fallback = dim3(2, 1, 1);
    RasterOrderOptions raster_order = RasterOrderOptions::AlongM;
    std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;

    problem_sizes_host.reserve(E);
    for (int i = 0; i < E; i++) {
        const uint32 start = expert_offsets[i].item<int64>();
        const uint32 end = i == E - 1 ? TK : expert_offsets[i + 1].item<int64>();
        const uint32 M = end - start;
        problem_sizes_host.push_back({M, N, K});
    }

    allocate(problem_sizes_host);
    initialize(alpha, beta, problem_sizes_host);

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
    const bool passed = verify(alpha, beta, problem_sizes_host);

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
        double gflops = get_gflops(double(timer.elapsed_millis()) / double(iterations) / 1000.0, problem_sizes_host);
        std::cout << "  TFLOPS      : " << gflops / 1000.0 << std::endl;
    }
}
