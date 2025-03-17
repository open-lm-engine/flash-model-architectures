#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "include/activations.h"
#include "include/dtypes/dtypes.h"
#include "include/launch.h"
#include "include/math.h"
#include "include/threads.h"

namespace ck = cute_kernels;

using uint32 = ck::uint32;
using fp32 = ck::fp32;
using fp32_2 = ck::fp32_2;
using fp32_4 = ck::fp32_4;

template <typename scalar_t>
__global__ void _swiglu_forward_cuda_kernel(const scalar_t *gate,
                                            const scalar_t *up,
                                            scalar_t *output,
                                            const uint32 num_elements) {
    constexpr int num_elements_per_thread = 16 / sizeof(scalar_t);
    static_assert(num_elements_per_thread == 4 || num_elements_per_thread == 8);

    using dtype = ck::DType<scalar_t>;

    const uint32 thread_id = ck::get_global_thread_id();
    const uint32 num_elements4 = num_elements / num_elements_per_thread;

    if (thread_id < num_elements4) {
        const fp32 *gate_vec = (fp32 *)&((fp32_4 *)gate)[thread_id];
        const fp32 *up_vec = (fp32 *)&((fp32_4 *)up)[thread_id];
        fp32 output_buffer[4];

        // clang-format off
        #pragma unroll
        // clang-format on
        for (int i = 0; i < 4; i++) {
            if constexpr (std::is_same_v<scalar_t, fp32>) {
                output_buffer[i] = up_vec[i] * gate_vec[i] * ck::sigmoid<fp32, fp32>(gate_vec[i]);
            } else {
                fp32_2 _gate_upcast = dtype::upcast(dtype::reinterpret_32_bits_as_2x16(gate_vec[i]));
                fp32_2 _up_upcast = dtype::upcast(dtype::reinterpret_32_bits_as_2x16(up_vec[i]));

                _gate_upcast =
                    ck::DType<fp32>::make2(_up_upcast.x * _gate_upcast.x * ck::sigmoid<fp32, fp32>(_gate_upcast.x),
                                           _up_upcast.y * _gate_upcast.y * ck::sigmoid<fp32, fp32>(_gate_upcast.y));

                output_buffer[i] = dtype::reinterpret_2x16_as_32_bits(dtype::downcast(_gate_upcast));
            }
        }

        ((fp32_4 *)output)[thread_id] = ck::DType<fp32>::make4(output_buffer);
    }

    const uint32 index = num_elements4 * num_elements_per_thread + thread_id;
    if (index < num_elements) {
        fp32 _gate_upcast = dtype::upcast(gate[index]);

        // up is upcasted automatically
        _gate_upcast = up[index] * _gate_upcast * ck::sigmoid<fp32, fp32>(_gate_upcast);
        output[index] = dtype::downcast(_gate_upcast);
    }
}

void swiglu_forward_cuda(const torch::Tensor &gate,
                         const torch::Tensor &up,
                         torch::Tensor &output,
                         const uint32 &BLOCK_SIZE) {
    TORCH_CHECK(gate.is_cuda());
    TORCH_CHECK(up.is_cuda());
    TORCH_CHECK(output.is_cuda());

    TORCH_CHECK(BLOCK_SIZE % WARP_SIZE == 0);

    const uint64 total_elements = gate.numel();

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(gate.scalar_type(), "swiglu_forward_cuda_kernel", ([&] {
                                       const uint32 num_elements_per_thread = 16 / sizeof(scalar_t);
                                       const uint32 num_elements_per_block = BLOCK_SIZE * num_elements_per_thread;

                                       std::vector<ck::ChunkedArray<scalar_t>> gate_chunks =
                                           ck::chunk_array<scalar_t>(gate.data_ptr<scalar_t>(), total_elements);
                                       std::vector<ck::ChunkedArray<scalar_t>> up_chunks =
                                           ck::chunk_array<scalar_t>(up.data_ptr<scalar_t>(), total_elements);
                                       std::vector<ck::ChunkedArray<scalar_t>> output_chunks =
                                           ck::chunk_array<scalar_t>(output.data_ptr<scalar_t>(), total_elements);

                                       for (int i = 0; i < gate_chunks.size(); i++) {
                                           ck::ChunkedArray<scalar_t> gate_chunk = gate_chunks[i];
                                           ck::ChunkedArray<scalar_t> up_chunk = up_chunks[i];
                                           ck::ChunkedArray<scalar_t> output_chunk = output_chunks[i];

                                           const uint32 num_elements = gate_chunk.num_elements;
                                           const uint32 NUM_BLOCKS =
                                               ck::ceil_divide<uint64>(num_elements, num_elements_per_block);

                                           _swiglu_forward_cuda_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                                               gate_chunk.array, up_chunk.array, output_chunk.array, num_elements);
                                       }
                                   }));
}
