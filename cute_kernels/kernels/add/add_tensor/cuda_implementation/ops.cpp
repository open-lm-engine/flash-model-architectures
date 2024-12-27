#include <torch/extension.h>

#include "../../../../include/dtypes/all.h"

void add_tensor_forward_cuda(const torch::Tensor &x,
                             const torch::Tensor &y,
                             torch::Tensor &output,
                             const int &vector_instruction_width,
                             const int &BLOCK_SIZE);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_tensor_forward_cuda", &add_tensor_forward_cuda, "Tensor addition forward (CUDA)");
}
