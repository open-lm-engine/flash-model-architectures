#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "common.h"

template <>
struct DType<uint32> {
    using nv_dtype = uint32;
    using nv_dtype2 = uint32_2;
    using nv_dtype4 = uint32_4;

    inline __device__ static nv_dtype2 make2(const nv_dtype &x, const nv_dtype &y) { return make_uint2(x, y); }
    inline __device__ static nv_dtype4 make4(const nv_dtype &x, const nv_dtype &y, const nv_dtype &z, const nv_dtype &t) { return make_uint4(x, y, z, t); }
};
