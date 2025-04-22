#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "include/cute_kernels.h"

namespace ck = cute_kernels;
namespace ck_mem = ck::memory;

using fp32 = ck::fp32;
using uint32 = ck::uint32;
using uint64 = ck::uint64;

template <typename integer_t>
inline __device__ void _load_cu_seqlens(const integer_t *cu_seqlens, integer_t *cu_seqlens_shared, const uint32 &B) {
    constexpr uint32 num_elements_per_thread = sizeof(integer_t);
    const uint32 B4 = B / num_elements_per_thread;

    for (uint32 i = threadIdx.x; i < B4; i += blockDim.x) {
        const uint32 index = i * num_elements_per_thread;
        uint32 *cu_seqlens_loaded = ck_mem::load_128_bits<integer_t>(cu_seqlens, i);

        for (uint32 j = 0; j < num_elements_per_thread; j++) {
            cu_seqlens_shared[index + j] = cu_seqlens_loaded[j];
        }
    }

    // use first warp to load remaining elements
    const uint32 index = (B4 * num_elements_per_thread) + threadIdx.x;
    if (index < B) {
        cu_seqlens_shared[index] = cu_seqlens[index];
    }
}

template <typename scalar_t, typename integer_t>
__global__ void _pack_sequence_cuda_kernel(const scalar_t *x,
                                           scalar_t *output,
                                           const uint32 *cu_seqlens,
                                           const uint32 *max_seqlen_tensor,
                                           const uint32 max_seqlen,
                                           const uint32 B,
                                           const uint32 S) {
    __shared__ integer_t max_seqlen_shared;
    __shared__ integer_t cu_seqlens_shared[B];

    _load_cu_seqlens<integer_t>(cu_seqlens, cu_seqlens_shared, B);

    // load max_seqlen into shared memory using 1st thread of each threadblock
    if (threadIdx.x == 0) {
        max_seqlen_shared = max_seqlen_tensor[0];
    }

    __syncthreads();
}

void pack_sequence_cuda(const torch::Tensor &x,
                        torch::Tensor &output,
                        const torch::Tensor &cu_seqlens,
                        const std::optional<torch::Tensor> &max_seqlen_tensor,
                        const std::optional<uint32> &max_seqlen,
                        const string &padding_side,
                        const uint32 &BLOCK_SIZE) {
    CHECK_CUDA_TENSOR(x);
    CHECK_CUDA_TENSOR(output);
    CHECK_CUDA_TENSOR(cu_seqlens);
    CHECK_CUDA_TENSOR(max_seqlen_tensor);

    CHECK_VALID_THREAD_BLOCK(BLOCK_SIZE);

    TORCH_CHECK(padding_side == "left" || padding_side == "right");

    // only one of the 2 should contain a value
    if (max_seqlen_tensor.has_value()) {
        TORCH_CHECK(!max_seqlen.has_value());
    } else {
        TORCH_CHECK(max_seqlen.has_value());
    }

    const uint64 num_elements = x.numel();
    // FIXME check this value
    TORCH_CHECK(num_elements < 1000000000);

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(
        x.scalar_type(), "pack_sequence_cuda_kernel", ([&] {
            constexpr uint32 num_elements_per_thread = ck_mem::get_num_elements_for_vector_load_stores<scalar_t>();
            const uint32 num_elements_per_block = num_elements_per_thread * BLOCK_SIZE;

            TORCH_CHECK(num_elements % num_elements_per_thread == 0);

            const uint32 NUM_BLOCKS = ck::ceil_divide<uint64>(num_elements, num_elements_per_block);

            _pack_sequence_cuda_kernel<scalar_t, uint32><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                x.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                cu_seqlens.data_ptr<uint32>(),
                max_seqlen_tensor.has_value() ? max_seqlen_tensor.value().data_ptr<uint32>() : nullptr,
                max_seqlen.has_value() ? max_seqlen.value() : nullptr,
                B,
                S);
        }));
}
