// **************************************************
// Copyright (c) 2025, Mayank Mishra
// **************************************************

#include <torch/extension.h>

void oops();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("oops", &oops, "oops GEMM (CUDA)"); }
