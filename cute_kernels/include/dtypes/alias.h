#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace cute_kernels {
#define _AT_DISPATCH_CASE_CUSTOM_FLOAT_TYPES(...)           \
    AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)     \
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)

#define AT_DISPATCH_CUSTOM_FLOAT_TYPES(TYPE, NAME, ...) \
    AT_DISPATCH_SWITCH(TYPE, NAME, _AT_DISPATCH_CASE_CUSTOM_FLOAT_TYPES(__VA_ARGS__))

#define _AT_DISPATCH_CASE_CUSTOM_INT_TYPES(...)        \
    AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)

#define AT_DISPATCH_CUSTOM_INT_TYPES(TYPE, NAME, ...) \
    AT_DISPATCH_SWITCH(TYPE, NAME, _AT_DISPATCH_CASE_CUSTOM_INT_TYPES(__VA_ARGS__))

    // define dtype aliases
    using fp64 = double;
    using fp64_2 = double2;
    using fp64_4 = double4;

    using fp32 = float;
    using fp32_2 = float2;
    using fp32_4 = float4;

    using fp16 = half;
    using fp16_2 = half2;

    using bf16 = __nv_bfloat16;
    using bf16_2 = __nv_bfloat162;

    using int64 = long;
    using uint64 = ulong;
    using uint64_2 = ulong2;

    using int32 = int;
    using int32_4 = int4;

    using uint32 = uint;
    using uint32_2 = uint2;
    using uint32_4 = uint4;

    using int16 = short;
    using uint16 = ushort;
}  // namespace cute_kernels
