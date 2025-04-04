#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "include/cute_kernels.h"

namespace ck = cute_kernels;
namespace ck_mem = ck::memory;

using fp32 = ck::fp32;
using fp32_4 = ck::fp32_4;
using uint32 = ck::uint32;
using uint64 = ck::uint64;

template <typename T, typename vecT>
inline __device__ T *load_128_bits(T *array, const uint64 &index) {
    vecT *vector_array = reinterpret_cast<vecT *>(array);
    vecT vector_element = vector_array[index];
    return reinterpret_cast<T *>(vector_element);
}

template <typename T, typename vecT>
inline __device__ T *store_128_bits(T *source, T *destination, const uint64 &index) {
    vecT *destination_vector_array = reinterpret_cast<vecT *>(destination);
    vecT source_vector = reinterpret_cast<vecT>(source);
    destination_vector_array[index] = source_vector;
}

template <typename scalar_t>
__global__ void _add_tensor_cuda_kernel(const scalar_t *x,
                                        const scalar_t *y,
                                        scalar_t *output,
                                        const uint64 num_elements) {
    constexpr uint32 num_elements_per_thread = 16 / sizeof(scalar_t);
    constexpr uint32 increment = num_elements_per_thread / 4;

    const uint32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32 num_vector_elements = num_elements / num_elements_per_thread;

    if (thread_id < num_vector_elements) {
        const scalar_t *x_vec = load_128_bits<const scalar_t, const fp32_4>(x, thread_id);
        const scalar_t *y_vec = load_128_bits<const scalar_t, const fp32_4>(y, thread_id);
        scalar_t output_buffer[num_elements_per_thread];

        for (uint32 i = 0; i < num_elements_per_thread; i += increment) {
            if constexpr (std::is_same_v<scalar_t, fp32>) {
                output_buffer[i] = x_vec[i] + y_vec[i];
            } else {
                using dtype = ck::DType<scalar_t>;
                using T2 = typename dtype::nv_dtype2;

                const uint32 i1 = i + 1;
                T2 x2 = dtype::make2(x_vec[i], x_vec[i1]);
                T2 y2 = dtype::make2(y_vec[i], y_vec[i1]);
                x2 = __hadd2(x2, y2);

                output_buffer[i] = x2.x;
                output_buffer[i1] = x2.y;
            }
        }

        store_128_bits<scalar_t, fp32_4>(output_buffer, output, thread_id);
    }

    const uint32 index = num_vector_elements * num_elements_per_thread + thread_id;
    if (index < num_elements) {
        output[index] = x[index] + y[index];
    }
}

void add_tensor_cuda(const torch::Tensor &x, const torch::Tensor &y, torch::Tensor &output, const uint32 &BLOCK_SIZE) {
    CHECK_CUDA_TENSOR(x);
    CHECK_CUDA_TENSOR(y);
    CHECK_CUDA_TENSOR(output);

    CHECK_VALID_THREAD_BLOCK(BLOCK_SIZE);

    const uint64 total_elements = x.numel();

    AT_DISPATCH_CUSTOM_FLOAT_TYPES(x.scalar_type(), "add_tensor_cuda_kernel", ([&] {
                                       const uint32 num_elements_per_thread = 16 / sizeof(scalar_t);
                                       const uint32 num_elements_per_block = BLOCK_SIZE * num_elements_per_thread;

                                       std::vector<ck::ChunkedArray<scalar_t>> x_chunks =
                                           ck::chunk_array<scalar_t>(x.data_ptr<scalar_t>(), total_elements);
                                       std::vector<ck::ChunkedArray<scalar_t>> y_chunks =
                                           ck::chunk_array<scalar_t>(y.data_ptr<scalar_t>(), total_elements);
                                       std::vector<ck::ChunkedArray<scalar_t>> output_chunks =
                                           ck::chunk_array<scalar_t>(output.data_ptr<scalar_t>(), total_elements);

                                       for (int i = 0; i < x_chunks.size(); i++) {
                                           ck::ChunkedArray<scalar_t> x_chunk = x_chunks[i];
                                           ck::ChunkedArray<scalar_t> y_chunk = y_chunks[i];
                                           ck::ChunkedArray<scalar_t> output_chunk = output_chunks[i];

                                           const uint64 num_elements = x_chunk.num_elements;
                                           const uint32 NUM_BLOCKS =
                                               ck::ceil_divide<uint64>(num_elements, num_elements_per_block);

                                           _add_tensor_cuda_kernel<scalar_t><<<NUM_BLOCKS, BLOCK_SIZE>>>(
                                               x_chunk.array, y_chunk.array, output_chunk.array, num_elements);
                                       }
                                   }));
}
