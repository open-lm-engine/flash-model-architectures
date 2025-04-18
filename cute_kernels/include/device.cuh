#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>

#include "dtypes.h"

namespace cute_kernels {
    inline __host__ uint32 get_num_SMs() {
        int32 device_id;
        cudaGetDevice(&device_id);

        cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, device_id);

        return properties.multiProcessorCount;
    }
}  // namespace cute_kernels
