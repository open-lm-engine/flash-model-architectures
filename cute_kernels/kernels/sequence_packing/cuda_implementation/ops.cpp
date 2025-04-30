#include <torch/extension.h>

void pack_sequence_cuda(const torch::Tensor &x,
                        torch::Tensor &output,
                        const torch::Tensor &cu_seqlens,
                        const std::string &padding_side,
                        const uint &BLOCK_SIZE);

void unpack_sequence_cuda(const torch::Tensor &x,
                          torch::Tensor &output,
                          const torch::Tensor &cu_seqlens,
                          const std::optional<torch::Tensor> &max_seqlen_tensor,
                          const std::optional<uint> &max_seqlen,
                          const std::string &padding_side,
                          const uint &BLOCK_SIZE);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_sequence_cuda", &pack_sequence_cuda, "pack sequence (CUDA)");
    m.def("unpack_sequence_cuda", &unpack_sequence_cuda, "unpack sequence (CUDA)");
}
