#include "../dtypes.h"

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
