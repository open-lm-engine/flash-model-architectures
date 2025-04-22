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

template <typename scalar_t, typename integer_t, bool has_trailing_elements>
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
                        std::optional<const torch::Tensor> &max_seqlen_tensor,
                        std::optional<const uint32> &max_seqlen,
                        const uint32 &BLOCK_SIZE) {
    CHECK_CUDA_TENSOR(x);
    CHECK_CUDA_TENSOR(output);
    CHECK_CUDA_TENSOR(cu_seqlens);
    CHECK_CUDA_TENSOR(max_seqlen);

    CHECK_VALID_THREAD_BLOCK(BLOCK_SIZE);

    // only one of the 2 should contain a value
    if (max_seqlen_tensor.has_value()) {
        TORCH_CHECK(!max_seqlen.has_value());
    } else {
        TORCH_CHECK(max_seqlen.has_value());
    }

    const uint64 total_elements = x.numel();

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(
        x.scalar_type(), "pack_sequence_cuda_kernel", ([&] {
            const uint32 num_elements_per_thread = 16 / sizeof(scalar_t);
            const uint32 num_elements_per_block = num_elements_per_thread * BLOCK_SIZE;

            std::vector<ck::ChunkedArray<scalar_t>> x_chunks =
                ck::chunk_array<scalar_t>(x.data_ptr<scalar_t>(), total_elements);
            std::vector<ck::ChunkedArray<scalar_t>> output_chunks =
                ck::chunk_array<scalar_t>(output.data_ptr<scalar_t>(), total_elements);

            for (int i = 0; i < x_chunks.size(); i++) {
                ck::ChunkedArray<scalar_t> x_chunk = x_chunks[i];
                ck::ChunkedArray<scalar_t> y_chunk = y_chunks[i];
                ck::ChunkedArray<scalar_t> output_chunk = output_chunks[i];

                const uint64 num_elements = x_chunk.num_elements;
                const bool has_trailing_elements =
                    (i == x_chunks.size() - 1) && (num_elements % num_elements_per_thread != 0);

                if (has_trailing_elements) {
                    const uint32 num_elements_per_warp = num_elements_per_thread << LOG_WARP_SIZE;
                    const uint32 num_warps_per_block = BLOCK_SIZE >> LOG_WARP_SIZE;
                    // 1 extra warp to avoid thread divergence
                    const uint32 NUM_WARPS = ck::ceil_divide<uint64>(num_elements, num_elements_per_warp) + 1;
                    const uint32 NUM_BLOCKS = ck::ceil_divide<uint64>(NUM_WARPS, num_warps_per_block);

                    _pack_sequence_cuda_kernel<scalar_t, uint32, true>
                        <<<NUM_BLOCKS, BLOCK_SIZE>>>(x_chunk.array, y_chunk.array, output_chunk.array, num_elements);
                } else {
                    const uint32 NUM_BLOCKS = ck::ceil_divide<uint64>(num_elements, num_elements_per_block);

                    _pack_sequence_cuda_kernel<scalar_t, uint32, false>
                        <<<NUM_BLOCKS, BLOCK_SIZE>>>(x_chunk.array, y_chunk.array, output_chunk.array, num_elements);
                }
            }
        }));
}
