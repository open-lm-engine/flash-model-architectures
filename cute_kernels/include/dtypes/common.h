// **************************************************
// Copyright (c) 2025, Mayank Mishra
// **************************************************

#pragma once

#include "alias.h"

namespace fma {
    // base struct for converting torch ScalarType to NVIDIA's dtype
    template <typename scalar_t>
    struct DType {
        using c10_dtype = scalar_t;
    };
}  // namespace fma
