#include <torch/extension.h>
#include "flash_attn_cuda.h"

torch::Tensor flash_attn_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value
) {
    TORCH_CHECK(query.is_cuda(), "query must be CUDA");
    TORCH_CHECK(query.scalar_type() == at::kHalf, "query must be fp16");

    auto output = torch::zeros_like(query);

    // call CUDA launcher (NORMAL FUNCTION)
    flash_attn_forward_cuda(query, key, value, output);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_forward", &flash_attn_forward, "FlashAttention forward");
}