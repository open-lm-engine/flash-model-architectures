#include <torch/extension.h>

void add_tensor_cuda(const torch::Tensor &x, const torch::Tensor &y, torch::Tensor &output, const uint &BLOCK_SIZE);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("add_tensor_cuda", &add_tensor_cuda, "Tensor addition (CUDA)"); }
