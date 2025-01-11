#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "../../../include/activations.h"
#include "../../../include/dtypes/all.h"
#include "../../../include/launch.h"
#include "../../../include/math.h"
#include "../../../include/threads.h"

template <typename scalar_t>
__global__ void _swiglu_backward_cuda_kernel(const scalar_t *gate,
                                             const scalar_t *up,
                                             const scalar_t *output_grad,
                                             scalar_t *gate_grad,
                                             scalar_t *up_grad,
                                             const uint64 num_elements) {
    constexpr int vector_instruction_width = sizeof(fp32_4) / sizeof(scalar_t);
    static_assert(vector_instruction_width == 4 || vector_instruction_width == 8);

    const uint64 thread_id = get_global_thread_id();
    using dtype = DType<scalar_t>;

    uint64 end = (thread_id + 1) * vector_instruction_width - 1;  // inclusive of last element

    if (end < num_elements) {
        const fp32 *gate_vec = (fp32 *)&((fp32_4 *)gate)[thread_id];
        const fp32 *up_vec = (fp32 *)&((fp32_4 *)up)[thread_id];
        const fp32 *output_grad_vec = (fp32 *)&((fp32_4 *)output_grad)[thread_id];

        fp32 gate_grad_buffer[4];
        fp32 up_grad_buffer[4];

        // clang-format off
        #pragma unroll
        // clang-format on
        for (int i = 0; i < 4; i++) {
            if constexpr (std::is_same_v<scalar_t, fp32>) {
                fp32 _gate_sigmoid = sigmoid<fp32, fp32>(gate_vec[i]);
                fp32 _gate_silu = gate_vec[i] * _gate_sigmoid;

                gate_grad_buffer[i] =
                    output_grad_vec[i] * up_vec[i] * (_gate_sigmoid + _gate_silu * (1 - _gate_sigmoid));
                up_grad_buffer[i] = output_grad_vec[i] * _gate_silu;
            } else {
                fp32_2 _gate_upcast = dtype::upcast(dtype::reinterpret_32_bits_as_2x16(gate_vec[i]));
                fp32_2 _up_upcast = dtype::upcast(dtype::reinterpret_32_bits_as_2x16(up_vec[i]));
                fp32_2 _output_grad_upcast = dtype::upcast(dtype::reinterpret_32_bits_as_2x16(output_grad_vec[i]));

                fp32 _gate_sigmoid_x = sigmoid<fp32, fp32>(_gate_upcast.x);
                fp32 _gate_sigmoid_y = sigmoid<fp32, fp32>(_gate_upcast.y);

                fp32 _gate_silu_x = _gate_upcast.x * _gate_sigmoid_x;
                fp32 _gate_silu_y = _gate_upcast.y * _gate_sigmoid_y;

                _gate_upcast = DType<fp32>::make2(
                    _output_grad_upcast.x * _up_upcast.x * (_gate_sigmoid_x + _gate_silu_x * (1 - _gate_sigmoid_x)),
                    _output_grad_upcast.y * _up_upcast.y * (_gate_sigmoid_y + _gate_silu_y * (1 - _gate_sigmoid_y)));

                _up_upcast =
                    DType<fp32>::make2(_output_grad_upcast.x * _gate_silu_x, _output_grad_upcast.y * _gate_silu_y);

                gate_grad_buffer[i] = dtype::reinterpret_2x16_as_32_bits(dtype::downcast(_gate_upcast));
                up_grad_buffer[i] = dtype::reinterpret_2x16_as_32_bits(dtype::downcast(_up_upcast));
            }
        }

        ((fp32_4 *)gate_grad)[thread_id] = DType<fp32>::make4(gate_grad_buffer);
        ((fp32_4 *)up_grad)[thread_id] = DType<fp32>::make4(up_grad_buffer);
    }

    // use first warp for computing the last elements
    if (thread_id < WARP_SIZE) {
        // NOTE end is same as start since we don't use vector load stores here
        end = (num_elements / vector_instruction_width) * vector_instruction_width + thread_id;
        if (end < num_elements) {
            fp32 _gate_upcast = dtype::upcast(gate[end]);

            fp32 _gate_sigmoid = sigmoid<fp32, fp32>(_gate_upcast);
            fp32 _gate_silu = _gate_upcast * _gate_sigmoid;

            gate_grad[end] =
                dtype::downcast(output_grad[end] * up[end] * (_gate_sigmoid + _gate_silu * (1 - _gate_sigmoid)));
            up_grad[end] = dtype::downcast(output_grad[end] * _gate_silu);
        }
    }
}

void swiglu_backward_cuda(const torch::Tensor &gate,
                          const torch::Tensor &up,
                          const torch::Tensor &output_grad,
                          torch::Tensor &gate_grad,
                          torch::Tensor &up_grad,
                          const uint32 &BLOCK_SIZE) {
    const uint64 total_elements = gate.numel();

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(
        gate.scalar_type(), "swiglu_backward_cuda_kernel", ([&] {
            const uint32 vector_instruction_width = 16 / sizeof(scalar_t);

            std::vector<ChunkedArray<scalar_t>> gate_chunks =
                chunk_array<scalar_t>(gate.data_ptr<scalar_t>(), total_elements);
            std::vector<ChunkedArray<scalar_t>> up_chunks =
                chunk_array<scalar_t>(up.data_ptr<scalar_t>(), total_elements);
            std::vector<ChunkedArray<scalar_t>> output_grad_chunks =
                chunk_array<scalar_t>(output_grad.data_ptr<scalar_t>(), total_elements);
            std::vector<ChunkedArray<scalar_t>> gate_grad_chunks =
                chunk_array<scalar_t>(gate_grad.data_ptr<scalar_t>(), total_elements);
            std::vector<ChunkedArray<scalar_t>> up_grad_chunks =
                chunk_array<scalar_t>(up_grad.data_ptr<scalar_t>(), total_elements);

            for (int i = 0; i < gate_chunks.size(); i++) {
                ChunkedArray<scalar_t> gate_chunk = gate_chunks[i];
                ChunkedArray<scalar_t> up_chunk = up_chunks[i];
                ChunkedArray<scalar_t> output_grad_chunk = output_grad_chunks[i];
                ChunkedArray<scalar_t> gate_grad_chunk = gate_grad_chunks[i];
                ChunkedArray<scalar_t> up_grad_chunk = up_grad_chunks[i];

                const uint64 num_elements = gate_chunk.num_elements;

                const uint32 num_elements_per_block = BLOCK_SIZE * vector_instruction_width;
                const uint32 NUM_BLOCKS = ceil_divide<uint64>(num_elements, num_elements_per_block);

                _swiglu_backward_cuda_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(gate_chunk.array,
                                                                                   up_chunk.array,
                                                                                   output_grad_chunk.array,
                                                                                   gate_grad_chunk.array,
                                                                                   up_grad_chunk.array,
                                                                                   num_elements);
            }
        }));
}
