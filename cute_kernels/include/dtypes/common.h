#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "alias.h"

namespace cute_kernels {
    inline __device__ std::tuple<uint16, uint16> split_fp32_into_16_bits(const fp32 &value) {
        uint32 left_right_int = __float_as_uint(value);

        uint16 right_int = left_right_int & 0xFFFF;
        uint16 left_int = left_right_int >> 16;

        return std::make_tuple(left_int, right_int);
    }

    inline __device__ fp32 combine_16_bits_into_fp32(const uint16 &left_int, const uint16 &right_int) {
        uint32 left_right_int = (static_cast<uint32>(left_int) << 16) | right_int;
        return __uint_as_float(left_right_int);
    }

    inline __device__ fp64 combine_32_bits_into_fp64(const uint32 &left_int, const uint32 &right_int) {
        uint64 left_right_int = (static_cast<uint64>(left_int) << 32) | right_int;
        return __longlong_as_double(left_right_int);
    }

    inline __device__ fp64 combine_16_bits_into_fp64(const uint16 &first_int,
                                                     const uint16 &second_int,
                                                     const uint16 &third_int,
                                                     const uint16 &fourth_int) {
        uint64 left_right_int = (static_cast<uint64>(first_int) << 48) | (static_cast<uint64>(second_int) << 32) |
                                (static_cast<uint64>(third_int) << 16) | fourth_int;
        return __longlong_as_double(left_right_int);
    }

    // base struct for converting torch ScalarType to NVIDIA's dtype
    template <typename scalar_t>
    struct DType {
        using c10_dtype = scalar_t;
    };
}  // namespace cute_kernels
