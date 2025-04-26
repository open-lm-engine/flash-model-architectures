#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "include/cute_kernels.h"

namespace ck = cute_kernels;
namespace ck_mem = ck::memory;

using fp32 = ck::fp32;
using uint32 = ck::uint32;
using uint64 = ck::uint64;

enum class PaddingSide { left, right };

template <typename scalar_t>
__device__ void _copy_array(const scalar_t *source,
                            scalar_t *destination,
                            const uint32 &b,
                            const uint32 &s,
                            const uint32 &output_index,
                            const uint32 &S,
                            const uint32 &N) {
    constexpr uint32 N_per_thread = ck_mem::get_num_elements_for_vector_load_stores<scalar_t>();
    const uint32 N_vec = N / N_per_thread;

    // start = b * stride_b + s * stride_s for N_per_thread = 1
    // start = (b * stride_b + s * stride_s) / N_per_thread for N_per_thread != 1
    uint32 load_offset = (b * S + s) * N_vec;
    uint32 store_offset = output_index * N_vec;

    for (uint32 i = threadIdx.x; i < N_vec; i += blockDim.x) {
        const scalar_t *source_vec = ck_mem::load_128_bits<const scalar_t>(source, load_offset + i);
        ck_mem::store_128_bits<scalar_t>(source_vec, destination, store_offset + i);
    }

    load_offset += N_vec;
    load_offset *= N_per_thread;

    store_offset += N_vec;
    store_offset *= N_per_thread;

    if (threadIdx.x < N) {
        destination[store_offset + threadIdx.x] = source[load_offset + threadIdx.x];
    }
}

template <typename scalar_t, typename integer_t, bool is_max_seqlen_tensor, PaddingSide padding_side>
__global__ void pack_sequence_cuda_kernel(const scalar_t *x,
                                          scalar_t *output,
                                          const uint32 *cu_seqlens,
                                          const uint32 *max_seqlen_tensor,
                                          uint32 max_seqlen,  // not constant to be able to laod from max_seqlen_tensor
                                          const uint32 B,
                                          const uint32 S,
                                          const uint32 N) {
    const uint32 s = blockIdx.x;
    const uint32 b = blockIdx.y;

    const uint32 start = cu_seqlens[b];
    const uint32 end = cu_seqlens[b + 1];
    const uint32 seqlens = end - start;

    if (padding_side == PaddingSide::left) {
        const uint32 pad_tokens = S - seqlens;

        if (s >= pad_tokens) {
            _copy_array(x, output, b, s, start + s - pad_tokens, S, N);
        }
    }
}

void pack_sequence_cuda(const torch::Tensor &x,
                        torch::Tensor &output,
                        const torch::Tensor &cu_seqlens,
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
                pack_sequence_cuda_kernel<scalar_t, uint32, true, PaddingSide::left>
                    <<<NUM_BLOCKS, BLOCK_SIZE, shared_memory_size>>>(x.data_ptr<scalar_t>(),
                                                                     output.data_ptr<scalar_t>(),
                                                                     seqlens.data_ptr<uint32>(),
                                                                     max_seqlen_tensor.value().data_ptr<uint32>(),
                                                                     0,
                                                                     B,
                                                                     S,
                                                                     N);
            } else {
                pack_sequence_cuda_kernel<scalar_t, uint32, false, PaddingSide::left>
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
