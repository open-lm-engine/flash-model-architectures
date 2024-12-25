#pragma once

#include <cuda.h>

__host__ int get_sm_count() {
    // get device id
    int device_id;
    cudaGetDevice(&device_id);

    // get device properties
    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, device_id);

    return device_properties.multiProcessorCount;
}
