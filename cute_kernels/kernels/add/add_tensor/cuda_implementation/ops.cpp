#include <torch/extension.h>

void add_tensor_forward_cuda(const torch::Tensor &x,
                             const torch::Tensor &y,
                             torch::Tensor &output,
                             const uint &vector_instruction_width,
                             const uint &BLOCK_SIZE);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_tensor_forward_cuda", &add_tensor_forward_cuda, "Tensor addition forward (CUDA)");
}
