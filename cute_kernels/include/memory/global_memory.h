#include "../dtypes.h"

#define CHECK_POWER_OF_2(x) static_assert(bits == 128 || bits == 64 || bits == 32)

namespace cute_kernels::memory {
    template <typename T, uint32 bits = 128>
    inline __device__ T *vectorized_load(T *array, const uint64 &index) {
        CHECK_POWER_OF_2(bits);

        using V = std::conditional_t<bits == 128, int32_4, std::conditional_t<bits == 64, int32_2, int32>>;
        using vecT = std::conditional_t<std::is_const<T>::value, const V, V>;

        vecT *vector_array = reinterpret_cast<vecT *>(array);
        vecT vector_element = vector_array[index];
        T *output = reinterpret_cast<T *>(&vector_element);

        return output;
    }

    template <typename T, uint32 bits = 128>
    inline __device__ void vectorized_store(T *source, T *destination, const uint64 &index) {
        CHECK_POWER_OF_2(bits);

        using V = std::conditional_t<bits == 128, int32_4, std::conditional_t<bits == 64, int32_2, int32>>;
        using vecT = std::conditional_t<std::is_const<T>::value, const V, V>;

        vecT *destination_vector_array = reinterpret_cast<vecT *>(destination);
        vecT source_vector = reinterpret_cast<vecT *>(&source[0])[0];
        destination_vector_array[index] = source_vector;
    }

    template <typename T, uint32 bits>
    constexpr inline __device__ uint32 get_num_elements_for_vector_load_stores() {
        CHECK_POWER_OF_2(bits);
        return (bits >> 3) / sizeof(T);
    }
}  // namespace cute_kernels::memory
