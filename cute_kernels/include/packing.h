// NOTE this file is copied from llm.c and is a good template for vector load/stores
// and is a nice alternative to manually casting everything to int4, loading and casting back

#include "dtypes.h"

// ----------------------------------------------------------------------------
// Packed128 data structure that forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.

template <class dtype>
struct alignas(16) Packed128 {
    Packed128() = default;

    static constexpr const uint32 size = sizeof(int32_4) / sizeof(dtype);
    dtype payload[size];

    __device__ explicit Packed128(int4 bits) {
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&payload, &bits, sizeof(bits));
    }

    __device__ static Packed128 constant(dtype value) {
        Packed128 result;
        for (int k = 0; k < size; ++k) {
            result.payload[k] = value;
        }
        return result;
    }

    __device__ static Packed128 zeros() { return constant(0.f); }
    __device__ static Packed128 ones() { return constant(1.f); }

    __device__ dtype& operator[](int index) { return payload[index]; }
    __device__ const dtype& operator[](int index) const { return payload[index]; }
    __device__ int4 get_bits() const {
        int4 bits;
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
        memcpy(&bits, &payload, sizeof(bits));
        return bits;
    }
};

// load a Packed128 from an aligned memory address
template <class dtype>
__device__ Packed128<dtype> load128(const dtype* address) {
    return Packed128<dtype>{*reinterpret_cast<const int4*>(address)};
}

// load a Packed128 from an aligned memory address with streaming cache hint
template <class dtype>
__device__ Packed128<dtype> load128cs(const dtype* address) {
    return Packed128<dtype>{__ldcs(reinterpret_cast<const int4*>(address))};
}

// store a Packed128 to an aligned memory address
template <class dtype>
__device__ void store128(dtype* target, Packed128<dtype> value) {
    *reinterpret_cast<int4*>(target) = value.get_bits();
}

// store a Packed128 to an aligned memory address with streaming cache hint
template <class dtype>
__device__ void store128cs(dtype* target, Packed128<dtype> value) {
    __stcs(reinterpret_cast<int4*>(target), value.get_bits());
}

// store a Packed128 to an aligned memory address while caching in L2 but bypassing L1
template <class dtype>
__device__ void store128cg(dtype* target, Packed128<dtype> value) {
    __stcg(reinterpret_cast<int4*>(target), value.get_bits());
}
