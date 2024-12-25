#include <torch/extension.h>

void contiguous_count_cuda(const torch::Tensor &x,
                           torch::Tensor &output,
                           const int &sm_count,
                           const int &thread_block_cluster_size,
                           const int &C,
                           const int &BLOCK_SIZE);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("contiguous_count_cuda", &contiguous_count_cuda, "contiguous count forward (CUDA)");
}
