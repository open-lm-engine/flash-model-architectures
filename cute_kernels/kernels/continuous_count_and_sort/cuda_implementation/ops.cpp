#include <torch/extension.h>

void continuous_count_and_sort_cuda(
    const torch::Tensor &x, torch::Tensor &count, const uint &sm_count, const uint &C, const uint &BLOCK_SIZE);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("continuous_count_and_sort_cuda", &continuous_count_and_sort_cuda, "contiguous count and sort (CUDA)");
}
