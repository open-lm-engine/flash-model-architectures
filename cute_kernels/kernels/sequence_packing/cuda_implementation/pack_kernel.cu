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
inline __device__ void _load_seqlens(const integer_t *seqlens, integer_t *seqlens_shared, const uint32 &B) {
    constexpr uint32 num_elements_per_thread = ck_mem::get_num_elements_for_vector_load_stores<integer_t>();
    const uint32 B_vec = B / num_elements_per_thread;

    for (uint32 i = threadIdx.x; i < B_vec; i += blockDim.x) {
        uint32 *cu_seqlens_loaded = ck_mem::load_128_bits<integer_t>(seqlens, i);

        for (uint32 j = 0; j < num_elements_per_thread; j++) {
            seqlens_shared[i * num_elements_per_thread + j] = cu_seqlens_loaded[j];
        }
    }

    // use first warp to load remaining elements
    const uint32 index = B_vec * num_elements_per_thread + threadIdx.x;
    if (index < B) {
        seqlens_shared[index] = seqlens[index];
    }
}

template <typename scalar_t, typename integer_t, bool is_max_seqlen_tensor, std::string padding_side>
__global__ void pack_sequence_cuda_kernel(const scalar_t *x,
                                          scalar_t *output,
                                          const uint32 *seqlens,
                                          const uint32 *max_seqlen_tensor,
                                          const uint32 max_seqlen,
                                          const uint32 B,
                                          const uint32 S,
                                          const uint32 N) {
    __shared__ integer_t max_seqlen_shared;
    integer_t *seqlens_shared = ck_mem::get_dynamic_shared_memory<integer_t>();

    // load max_seqlen into shared memory using 1st thread of each threadblock
    if (threadIdx.x == 0) {
        if (is_max_seqlen_tensor) {
            max_seqlen_shared = max_seqlen_tensor[0];
        } else {
            max_seqlen_shared = max_seqlen;
        }
    }

    _load_seqlens<integer_t>(seqlens, seqlens_shared, B);

    __syncthreads();

    const uint32 s = blockIdx.x;
    const uint32 b = blockIdx.y;
    const uint32 seqlens_b = seqlens_shared[b];

    bool is_pad_token = false;
    if (padding_side == "left") {
        if (s < S - seqlens_b) {
            is_pad_token = true;
        }
    } else if (padding_side == "right") {
        if (s > seqlens_b) {
            is_pad_token = true;
        }
    }

    if (is_pad_token) {
        return;
    }

    for
}

void pack_sequence_cuda(const torch::Tensor &x,
                        torch::Tensor &output,
                        const torch::Tensor &seqlens,
                        const std::optional<torch::Tensor> &max_seqlen_tensor,
                        const std::optional<uint32> &max_seqlen,
                        const std::string &padding_side,
                        const uint32 &BLOCK_SIZE) {
    CHECK_CUDA_TENSOR(x);
    CHECK_CUDA_TENSOR(output);
    CHECK_CUDA_TENSOR(seqlens);

    CHECK_VALID_THREAD_BLOCK(BLOCK_SIZE);

    TORCH_CHECK(padding_side == "left" || padding_side == "right");

    // only one of the 2 should contain a value
    if (max_seqlen_tensor.has_value()) {
        CHECK_CUDA_TENSOR(max_seqlen_tensor.value());
        TORCH_CHECK(!max_seqlen.has_value());
    } else {
        TORCH_CHECK(max_seqlen.has_value());
    }

    const uint32 B = x.size(0);
    const uint32 S = x.size(1);
    const uint32 N = x.numel() / (B * S);

    const dim3 NUM_BLOCKS = dim3(S, B);
    const uint32 shared_memory_size = B * sizeof(uint32);

    const uint64 N = x.numel();
    // FIXME check this value
    TORCH_CHECK(N < 1000000000);

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(
        x.scalar_type(), "pack_sequence_cuda_kernel", ([&] {
            constexpr uint32 N_per_thread = ck_mem::get_num_elements_for_vector_load_stores<scalar_t>();
            TORCH_CHECK(num_elements % N_per_thread == 0);

            if (max_seqlen_tensor.has_value()) {
                pack_sequence_cuda_kernel<scalar_t, uint32, true>
                    <<<NUM_BLOCKS, BLOCK_SIZE, shared_memory_size>>>(x.data_ptr<scalar_t>(),
                                                                     output.data_ptr<scalar_t>(),
                                                                     seqlens.data_ptr<uint32>(),
                                                                     max_seqlen_tensor.value().data_ptr<uint32>(),
                                                                     0,
                                                                     B,
                                                                     S,
                                                                     N);
            } else {
                pack_sequence_cuda_kernel<scalar_t, uint32, false>
                    <<<NUM_BLOCKS, BLOCK_SIZE, shared_memory_size>>>(x.data_ptr<scalar_t>(),
                                                                     output.data_ptr<scalar_t>(),
                                                                     seqlens.data_ptr<uint32>(),
                                                                     nullptr,
                                                                     max_seqlen.value(),
                                                                     B,
                                                                     S,
                                                                     N);
            }
        }));
}
