// **************************************************
// Copyright (c) 2025, Mayank Mishra
// **************************************************

#include <torch/extension.h>

void main();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("main", &main, "main GEMM (CUDA)"); }
