#pragma once
#include <torch/extension.h>

void flash_attn_forward_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor &output
);