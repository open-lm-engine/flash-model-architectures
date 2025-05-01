#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "include/cute_kernels.h"

namespace ck = cute_kernels;
namespace ck_mem = ck::memory;

using fp32 = ck::fp32;
using uint32 = ck::uint32;
using uint64 = ck::uint64;

template <typename scalar_t>
inline __device__ scalar_t _swiglu_forward(const scalar_t &gate, const scalar_t &up) {
    using dtype = ck::DType<scalar_t>;

    fp32 _up = dtype::upcast(up);
    fp32 _gate = dtype::upcast(gate);
    fp32 _sigmoid = ck::sigmoid<fp32, fp32>(_gate);

    _sigmoid *= _gate * _up;

    return dtype::downcast(_sigmoid);
}

template <typename scalar_t>
__global__ void swiglu_forward_cuda_kernel(const scalar_t *gate,
                                           const scalar_t *up,
                                           scalar_t *output,
                                           const uint64 N) {
    constexpr uint32 N_per_thread = ck_mem::get_num_elements_for_vector_load_stores<scalar_t>();

    const uint32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32 N_vec = N / N_per_thread;

    if (thread_id < N_vec) {
        const scalar_t *gate_vec = ck_mem::load_128_bits<scalar_t>(gate, thread_id);
        const scalar_t *up_vec = ck_mem::load_128_bits<scalar_t>(up, thread_id);
        scalar_t output_buffer[N_per_thread];

        for (uint32 i = 0; i < N_per_thread; i++) {
            output_buffer[i] = _swiglu_forward<scalar_t>(gate_vec[i], up_vec[i]);
        }

        ck_mem::store_128_bits<scalar_t>(output_buffer, output, thread_id);
    }

    const uint32 index = N_vec * N_per_thread + thread_id;
    if (index < N) {
        output[index] = _swiglu_forward<scalar_t>(gate[index], up[index]);
    }
}

void swiglu_forward_cuda(const torch::Tensor &gate,
                         const torch::Tensor &up,
                         torch::Tensor &output,
                         const uint32 &BLOCK_SIZE) {
    CHECK_CUDA_TENSOR(gate);
    CHECK_CUDA_TENSOR(up);
    CHECK_CUDA_TENSOR(output);

    CHECK_VALID_THREAD_BLOCK(BLOCK_SIZE);

    const uint64 total_elements = gate.numel();

    DISPATCH_FLOAT_KERNEL(gate.scalar_type(), "swiglu_forward_cuda_kernel", scalar_t, ([&] {
                              const uint32 N_per_thread = ck_mem::get_num_elements_for_vector_load_stores<scalar_t>();
                              const uint32 N_per_block = BLOCK_SIZE * N_per_thread;

                              std::vector<ck::ChunkedArray<scalar_t>> gate_chunks =
                                  ck::chunk_array<scalar_t>(gate.data_ptr<scalar_t>(), total_elements);
                              std::vector<ck::ChunkedArray<scalar_t>> up_chunks =
                                  ck::chunk_array<scalar_t>(up.data_ptr<scalar_t>(), total_elements);
                              std::vector<ck::ChunkedArray<scalar_t>> output_chunks =
                                  ck::chunk_array<scalar_t>(output.data_ptr<scalar_t>(), total_elements);

                              for (uint32 i = 0; i < gate_chunks.size(); i++) {
                                  ck::ChunkedArray<scalar_t> gate_chunk = gate_chunks[i];
                                  ck::ChunkedArray<scalar_t> up_chunk = up_chunks[i];
                                  ck::ChunkedArray<scalar_t> output_chunk = output_chunks[i];

                                  const uint64 N = gate_chunk.num_elements;
                                  const uint32 NUM_BLOCKS = ck::ceil_divide<uint64>(N, N_per_block);

                                  swiglu_forward_cuda_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                                      gate_chunk.array, up_chunk.array, output_chunk.array, N);
                              }
                          }));
}
