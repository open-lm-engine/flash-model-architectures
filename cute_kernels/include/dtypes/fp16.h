#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "alias.h"
#include "common.h"
#include "cutlass/half.h"

namespace cute_kernels {
    template <>
    struct DType<c10::Half> {
        using c10_dtype = c10::Half;
        using nv_dtype = fp16;
        using nv_dtype2 = fp16_2;
        using cutlass_dtype = cutlass::half_t;

        // fp32 -> fp16_2
        inline __device__ static nv_dtype2 reinterpret_32_bits_as_2x16(const fp32 &value) {
            auto [left_int, right_int] = split_fp32_into_16_bits(value);

            nv_dtype left = __ushort_as_half(left_int);
            nv_dtype right = __ushort_as_half(right_int);

            return __halves2half2(left, right);
        }

        // fp16_2 -> fp32
        inline __device__ static fp32 reinterpret_2x16_as_32_bits(const nv_dtype2 &value) {
            return reinterpret_2x16_as_32_bits(value.x, value.y);
        }

        // fp16, fp16 -> fp32
        inline __device__ static fp32 reinterpret_2x16_as_32_bits(const nv_dtype &left, const nv_dtype &right) {
            uint16 left_int = __half_as_ushort(left);
            uint16 right_int = __half_as_ushort(right);

            return combine_16_bits_into_fp32(left_int, right_int);
        }

        inline __device__ static fp32 upcast(const c10_dtype &value) { return upcast(static_cast<nv_dtype>(value)); }
        inline __device__ static fp32 upcast(const nv_dtype &value) { return __half2float(value); }
        inline __device__ static fp32_2 upcast(const nv_dtype2 &value) { return __half22float2(value); }

        inline __device__ static nv_dtype downcast(const fp32 &value) { return __float2half(value); }
        inline __device__ static nv_dtype2 downcast(const fp32_2 &value) { return __float22half2_rn(value); }

        inline __device__ static nv_dtype2 make2(const nv_dtype &value) { return __half2half2(value); }
        inline __device__ static nv_dtype2 make2(const nv_dtype &x, const nv_dtype &y) { return make_half2(x, y); }
        inline __device__ static nv_dtype2 make2(const nv_dtype *array) { return make_half2(array[0], array[1]); }
    };

    template <>
    struct DType<fp16> : public DType<c10::Half> {};
}  // namespace cute_kernels
