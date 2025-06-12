// **************************************************
// Copyright (c) 2025, Mayank Mishra
// **************************************************

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "include/cute_kernels.h"

namespace ck = cute_kernels;

using uint64 = ck::uint64;

void zeros_cuda(torch::Tensor &x) {
    CHECK_CUDA_TENSOR(x);

    const uint64 total_elements = x.numel();

    DISPATCH_FLOAT_KERNEL(x.scalar_type(), "zeros_cuda_kernel", scalar_t, ([&] {
                              const uint64 num_bytes = total_elements * sizeof(scalar_t);
                              cudaMemset(x.data_ptr<scalar_t>(), 0, num_bytes);  // zero all BF16s
                          }));
}
