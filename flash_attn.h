#pragma once
#include <torch/extension.h>
#include "flash_attn_cuda.h"

torch::Tensor flash_attn_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value
);