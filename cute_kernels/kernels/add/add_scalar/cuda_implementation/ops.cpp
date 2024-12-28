#include <torch/extension.h>

void add_scalar_forward_cuda(const torch::Tensor &x,
                             const float &y,
                             torch::Tensor &output,
                             const uint &vector_instruction_width,
                             const uint &BLOCK_SIZE);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_scalar_forward_cuda", &add_scalar_forward_cuda, "Scalar addition forward (CUDA)");
}
