// **************************************************
// Copyright (c) 2025, Mayank Mishra
// **************************************************

#include <torch/extension.h>

void zeros_cuda(torch::Tensor &x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("zeros_cuda", &zeros_cuda, "zeros (CUDA)"); }
