#include "../dtypes.h"

namespace cute_kernels::memory {
    template <typename T>
    inline __device__ T *load_128_bits(T *array, const uint64 &index) {
        using vecT = std::conditional_t<std::is_const<T>::value, const int32_4, int32_4>;
        vecT *vector_array = reinterpret_cast<vecT *>(array);
        vecT vector_element = vector_array[index];
        T *output = reinterpret_cast<T *>(&vector_element);
        return output;
    }

    template <typename sourceT, typename destinationT>
    inline __device__ void store_128_bits(sourceT *source, destinationT *destination, const uint64 &index) {
        static_assert(std::is_same_v<std::remove_const_t<sourceT>, destinationT>);

        using sourceV = std::conditional_t<std::is_const<sourceT>::value, const int32_4, int32_4>;
        using destinationV = std::conditional_t<std::is_const<destinationT>::value, const int32_4, int32_4>;

        destinationV *destination_vector_array = reinterpret_cast<destinationV *>(destination);
        destinationV source_vector = reinterpret_cast<destinationV *>(&source[0])[0];
        destination_vector_array[index] = source_vector;
    }

    template <typename T>
    constexpr inline __device__ uint32 get_num_elements_for_vector_load_stores() {
        return 16 / sizeof(T);
    }
}  // namespace cute_kernels::memory
