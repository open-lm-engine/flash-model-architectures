#pragma once

#include "dtypes/dtypes.h"

template <typename input_T, typename output_T>
__device__ output_T sigmoid(const input_T &x) {
    fp32 x_fp32 = DType<input_T>::upcast(x);

    if (x >= 0) {
        x_fp32 = 1 / (1 + expf(-x_fp32));
    } else {
        x_fp32 = expf(x_fp32);
        x_fp32 = x_fp32 / (1 + x_fp32);
    }

    return DType<output_T>::downcast(x_fp32);
}
