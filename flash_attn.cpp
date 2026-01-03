#include "flash_attn.h"

torch::Tensor flash_attn_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value
) {
    TORCH_CHECK(query.is_cuda(), "query must be CUDA");
    TORCH_CHECK(query.scalar_type() == at::kHalf, "query must be fp16");

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 blocks(
        (Lq + BLOCK_SIZE - 1) / BLOCK_SIZE, // tiles over query length
        H,                                  // heads
        B                                   // batch
    );

    int B  = query.size(0);
    int H  = query.size(1);
    int Lq = query.size(2);
    int Dh = query.size(3);
    int Lk = key.size(2);

    auto output = torch::zeros_like(query);

    flash_attention_kernel<<<blocks, threads>>>(
        reinterpret_cast<half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<half*>(key.data_ptr<at::Half>()),
        reinterpret_cast<half*>(value.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        Lq, Lk, H, Dh
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn_forward", &flash_attn_forward, "FlashAttention forward");
}