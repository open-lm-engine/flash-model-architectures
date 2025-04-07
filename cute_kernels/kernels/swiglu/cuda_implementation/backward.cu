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
inline __device__ void _swiglu_backward(const scalar_t &gate,
                                        const scalar_t &up,
                                        const scalar_t &output_grad,
                                        scalar_t *gate_grad_buffer,
                                        scalar_t *up_grad_buffer,
                                        const uint32 &index) {
    using dtype = ck::DType<scalar_t>;

    fp32 _gate = dtype::upcast(gate);
    fp32 _up = dtype::upcast(up);
    fp32 _output_grad = dtype::upcast(output_grad);

    fp32 _gate_sigmoid = ck::sigmoid<fp32, fp32>(_gate);
    fp32 _gate_silu = _gate * _gate_sigmoid;

    fp32 _gate_grad = _output_grad * _up * (_gate_sigmoid + _gate_silu * (1 - _gate_sigmoid));
    fp32 _up_grad = _output_grad * _gate_silu;

    scalar_t gate_grad = dtype::downcast(_gate_grad);
    scalar_t up_grad = dtype::downcast(_up_grad);

    gate_grad_buffer[index] = gate_grad;
    up_grad_buffer[index] = up_grad;
}

template <typename scalar_t>
__global__ void _swiglu_backward_cuda_kernel(const scalar_t *gate,
                                             const scalar_t *up,
                                             const scalar_t *output_grad,
                                             scalar_t *gate_grad,
                                             scalar_t *up_grad,
                                             const uint64 num_elements) {
    constexpr uint32 num_elements_per_thread = ck_mem::Packed128<scalar_t>::size;

    const uint32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32 num_vector_elements = num_elements / num_elements_per_thread;

    if (thread_id < num_vector_elements) {
        const scalar_t *gate_vec = ck_mem::load_128_bits<const scalar_t>(gate, thread_id);
        const scalar_t *up_vec = ck_mem::load_128_bits<const scalar_t>(up, thread_id);
        const scalar_t *output_grad_vec = ck_mem::load_128_bits<const scalar_t>(output_grad, thread_id);

        scalar_t gate_grad_buffer[num_elements_per_thread];
        scalar_t up_grad_buffer[num_elements_per_thread];

        for (uint32 i = 0; i < num_elements_per_thread; i++) {
            _swiglu_backward<scalar_t>(
                gate_vec[i], up_vec[i], output_grad_vec[i], gate_grad_buffer, up_grad_buffer, i);
        }

        ck_mem::Packed128Array<scalar_t> gate_grad_vec = ck_mem::Packed128Array<scalar_t>(gate_grad);
        gate_grad_vec[thread_id] = gate_grad_buffer;

        ck_mem::Packed128Array<scalar_t> up_grad_vec = ck_mem::Packed128Array<scalar_t>(up_grad);
        up_grad_vec[thread_id] = up_grad_buffer;
    }

    const uint32 index = num_vector_elements * num_elements_per_thread + thread_id;
    if (index < num_elements) {
        _swiglu_backward<scalar_t>(gate[index], up[index], output_grad[index], gate_grad, up_grad, index);
    }
}

void swiglu_backward_cuda(const torch::Tensor &gate,
                          const torch::Tensor &up,
                          const torch::Tensor &output_grad,
                          torch::Tensor &gate_grad,
                          torch::Tensor &up_grad,
                          const uint32 &BLOCK_SIZE) {
    CHECK_CUDA_TENSOR(gate);
    CHECK_CUDA_TENSOR(up);
    CHECK_CUDA_TENSOR(output_grad);
    CHECK_CUDA_TENSOR(gate_grad);
    CHECK_CUDA_TENSOR(up_grad);

    CHECK_VALID_THREAD_BLOCK(BLOCK_SIZE);

    const uint64 total_elements = gate.numel();

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(
        gate.scalar_type(), "swiglu_backward_cuda_kernel", ([&] {
            const uint32 num_elements_per_thread = 16 / sizeof(scalar_t);
            const uint32 num_elements_per_block = BLOCK_SIZE * num_elements_per_thread;

            std::vector<ck::ChunkedArray<scalar_t>> gate_chunks =
                ck::chunk_array<scalar_t>(gate.data_ptr<scalar_t>(), total_elements);
            std::vector<ck::ChunkedArray<scalar_t>> up_chunks =
                ck::chunk_array<scalar_t>(up.data_ptr<scalar_t>(), total_elements);
            std::vector<ck::ChunkedArray<scalar_t>> output_grad_chunks =
                ck::chunk_array<scalar_t>(output_grad.data_ptr<scalar_t>(), total_elements);
            std::vector<ck::ChunkedArray<scalar_t>> gate_grad_chunks =
                ck::chunk_array<scalar_t>(gate_grad.data_ptr<scalar_t>(), total_elements);
            std::vector<ck::ChunkedArray<scalar_t>> up_grad_chunks =
                ck::chunk_array<scalar_t>(up_grad.data_ptr<scalar_t>(), total_elements);

            for (int i = 0; i < gate_chunks.size(); i++) {
                ck::ChunkedArray<scalar_t> gate_chunk = gate_chunks[i];
                ck::ChunkedArray<scalar_t> up_chunk = up_chunks[i];
                ck::ChunkedArray<scalar_t> output_grad_chunk = output_grad_chunks[i];
                ck::ChunkedArray<scalar_t> gate_grad_chunk = gate_grad_chunks[i];
                ck::ChunkedArray<scalar_t> up_grad_chunk = up_grad_chunks[i];

                const uint64 num_elements = gate_chunk.num_elements;
                const uint32 NUM_BLOCKS = ck::ceil_divide<uint64>(num_elements, num_elements_per_block);

                _swiglu_backward_cuda_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(gate_chunk.array,
                                                                                   up_chunk.array,
                                                                                   output_grad_chunk.array,
                                                                                   gate_grad_chunk.array,
                                                                                   up_grad_chunk.array,
                                                                                   num_elements);
            }
        }));
}
