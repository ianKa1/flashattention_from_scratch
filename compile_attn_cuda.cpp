#include <cuda_runtime.h>
#include <iostream>
#include "flash_attn_cuda.cu"

int main() {
    flash_attn_forward(torch::randn(1, 8, 16, 128).cuda(), torch::randn(1, 8, 16, 128).cuda(), torch::randn(1, 8, 16, 128).cuda());
    return 0;
}