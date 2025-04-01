// NOTE this file is copied from llm.c and is a good template for vector load/stores
// and is a nice alternative to manually casting everything to int32_4, loading and casting back

#include "dtypes.h"

// Packed128 data structure that forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.

namespace cute_kernels::memory {
    template <typename T>
    struct alignas(16) Packed128 {
        Packed128() = default;

        static constexpr const uint32 size = sizeof(int32_4) / sizeof(T);
        T payload[size];

        inline __device__ explicit Packed128(int32_4& bits) {
            static_assert(sizeof(bits) == sizeof(payload), "size mismatch");
            memcpy(&payload, &bits, sizeof(bits));
        }

        inline __device__ static Packed128<T> constant(T& value) {
            Packed128 result;
            for (int i = 0; i < size; ++i) {
                result.payload[i] = value;
            }
            return result;
        }

        // Non-const accessors
        inline __device__ T& operator[](uint32& index) { return payload[index]; }
        inline __device__ T& operator[](const uint32& index) { return payload[index]; }
        // Const accessors
        inline __device__ const T& operator[](uint32& index) const { return payload[index]; }
        inline __device__ const T& operator[](const uint32& index) const { return payload[index]; }

        inline __device__ int32_4 get_bits() const {
            int32_4 bits;
            memcpy(&bits, &payload, sizeof(bits));
            return bits;
        }
    };

    template <typename T>
    struct alignas(16) Packed128Array {
        using packed_type = std::conditional_t<std::is_const<T>::value, const Packed128<T>, Packed128<T>>;
        packed_type* _array;

        inline __device__ explicit Packed128Array(T* array) { _array = reinterpret_cast<packed_type*>(array); }

        // Non-const accessors
        inline __device__ packed_type& operator[](uint64& index) { return _array[index]; }
        inline __device__ packed_type& operator[](const uint64& index) { return _array[index]; }
        // Const accessors
        inline __device__ const packed_type& operator[](uint64& index) const { return _array[index]; }
        inline __device__ const packed_type& operator[](const uint64& index) const { return _array[index]; }
    };
}  // namespace cute_kernels::memory
