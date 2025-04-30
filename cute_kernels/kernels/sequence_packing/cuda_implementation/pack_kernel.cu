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
inline __device__ void _copy_array(const scalar_t *source,
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
        const scalar_t *source_vec = ck_mem::load_128_bits<scalar_t>(source, load_offset + i);
        ck_mem::store_128_bits<scalar_t>(source_vec, destination, store_offset + i);
    }
}

template <typename scalar_t, typename integer_t, PaddingSide padding_side>
__global__ void pack_sequence_cuda_kernel(
    const scalar_t *x, scalar_t *output, const uint32 *cu_seqlens, const uint32 B, const uint32 S, const uint32 N) {
    const uint32 s = blockIdx.x;
    const uint32 b = blockIdx.y;

    const uint32 start = cu_seqlens[b];
    const uint32 end = cu_seqlens[b + 1];
    const uint32 seqlens = end - start;
    const uint32 pad_tokens = S - seqlens;

    if (padding_side == PaddingSide::left) {
        if (s >= pad_tokens) {
            _copy_array(x, output, b, s, start + s - pad_tokens, S, N);
        }
    } else {
        if (s < pad_tokens) {
            _copy_array(x, output, b, s, start + s, S, N);
        }
    }
}

void pack_sequence_cuda(const torch::Tensor &x,
                        torch::Tensor &output,
                        const torch::Tensor &cu_seqlens,
                        const std::string &padding_side,
                        const uint32 &BLOCK_SIZE) {
    CHECK_CUDA_TENSOR(x);
    CHECK_CUDA_TENSOR(output);
    CHECK_CUDA_TENSOR(cu_seqlens);

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

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(
        x.scalar_type(), "pack_sequence_cuda_kernel", ([&] {
            constexpr uint32 N_per_thread = ck_mem::get_num_elements_for_vector_load_stores<scalar_t>();
            TORCH_CHECK(N % N_per_thread == 0);

            if (padding_side == "left") {
                pack_sequence_cuda_kernel<scalar_t, uint32, PaddingSide::left>
                    <<<NUM_BLOCKS, BLOCK_SIZE, shared_memory_size>>>(
                        x.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), cu_seqlens.data_ptr<uint32>(), B, S, N);
            } else {
                pack_sequence_cuda_kernel<scalar_t, uint32, PaddingSide::right>
                    <<<NUM_BLOCKS, BLOCK_SIZE, shared_memory_size>>>(
                        x.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), cu_seqlens.data_ptr<uint32>(), B, S, N);
            }
        }));
}
