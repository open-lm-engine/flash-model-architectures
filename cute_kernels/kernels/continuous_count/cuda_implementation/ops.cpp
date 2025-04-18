#include <torch/extension.h>

void continuous_count_cuda(const torch::Tensor &x,
                           torch::Tensor &output,
                           const uint &thread_block_cluster_size,
                           const uint &C,
                           const uint &BLOCK_SIZE);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("continuous_count_cuda", &continuous_count_cuda, "contiguous count (CUDA)");
}
