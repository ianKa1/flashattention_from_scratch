#include <torch/extension.h>
#include <cuda_fp16.h>

__global__ void add_kernel(half* q, half* k, half* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[0] = __float2half(42.0f);
}

torch::Tensor flash_attn_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    // Ensure tensors are on the GPU
    assert(q.is_cuda());
    assert(k.is_cuda());
    assert(v.is_cuda());

    // Allocate output tensor
    auto out = torch::zeros_like(q);

    int B = q.size(0);
    int H = q.size(1);
    int L = q.size(2);
    int Dh = q.size(3);
    printf("B: %d, H: %d, L: %d, Dh: %d\n", B, H, L, Dh);
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < H; j++) {
            for (int l = 0; l < L; l++) {
                for (int d = 0; d < Dh; d++) {
                    printf("q[%d][%d][%d][%d]: %f\n", i, j, l, d, q[i][j][l][d].item<half>());
                }
            }
        }
    }

    // Kernel launch configuration
    int threads = 1024;
    int blocks = (q.size(0) + threads - 1) / threads;

    // Launch the kernel
    add_kernel<<<blocks, threads>>>(q.data_ptr<half>(), k.data_ptr<half>(), out.data_ptr<half>());

    // Return the result
    return out;
}