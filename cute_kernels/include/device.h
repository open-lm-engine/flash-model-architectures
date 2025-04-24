#include "dtypes.h"

namespace cute_kernels {
    inline __host__ uint32 get_num_SMs() {
        int32 device_id;
        cudaGetDevice(&device_id);

        int32 num_SMs;
        cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, device_id);

        return num_SMs;
    }
}  // namespace cute_kernels
