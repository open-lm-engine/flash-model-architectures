#pragma once

#include "alias.h"

namespace cute_kernels {
    // base struct for converting torch ScalarType to NVIDIA's dtype
    template <typename scalar_t>
    struct DType {
        using c10_dtype = scalar_t;
    };
}  // namespace cute_kernels
