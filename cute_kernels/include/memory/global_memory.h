// NOTE this file is copied from llm.c and is a good template for vector load/stores
// and is a nice alternative to manually casting everything to int32_4, loading and casting back

#include "../dtypes.h"

// Packed128 data structure that forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.

namespace cute_kernels::memory {
    template <typename T, typename vecT>
    inline __device__ T *load_128_bits(T *array, const uint64 &index) {
        vecT *vector_array = reinterpret_cast<vecT *>(array);
        vecT vector_element = vector_array[index];
        T *output = reinterpret_cast<T *>(&vector_element);
        return output;
    }

    template <typename T, typename vecT>
    inline __device__ void store_128_bits(T *source, T *destination, const uint64 &index) {
        vecT *destination_vector_array = reinterpret_cast<vecT *>(destination);
        vecT source_vector = reinterpret_cast<vecT *>(&source[0])[0];
        destination_vector_array[index] = source_vector;
    }

    template <typename T, typename vecT>
    constexpr inline __device__ uint32 get_num_elements_for_vector_load_stores() {
        return sizeof(vecT) / sizeof(T);
    }
}  // namespace cute_kernels::memory
