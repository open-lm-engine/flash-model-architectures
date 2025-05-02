#include <torch/extension.h>

void pack_unpack_sequence_cuda(const torch::Tensor &x,
                               torch::Tensor &output,
                               const torch::Tensor &cu_seqlens,
                               const std::string &padding_side,
                               const bool &pack,
                               const uint &BLOCK_SIZE);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_unpack_sequence_cuda", &pack_unpack_sequence_cuda, "pack unpack sequence (CUDA)");
}
