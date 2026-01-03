#include "flash_attn_cuda.h"
#include <cstdio>

#define BLOCK_SIZE 32

// CUDA kernel for FlashAttention
__global__ void flash_attention_kernel(half* query, half* key, half* value, half* out, int Lq, int Lk, int H, int Dh) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int head_id = blockIdx.y;  // For multi-head attention

    // Shared memory for each block
    // __shared__ half shared_q[BLOCK_SIZE][BLOCK_SIZE];
    // __shared__ half shared_k[BLOCK_SIZE][BLOCK_SIZE];
    // __shared__ half shared_v[BLOCK_SIZE][BLOCK_SIZE];
    
    if (idx < Lq) {
        // Load query, key, and value from global memory to shared memory
        // shared_q[threadIdx.x][threadIdx.y] = query[head_id * Lq * Dh + idx];
        // shared_k[threadIdx.x][threadIdx.y] = key[head_id * Lk * Dh + idx];
        // shared_v[threadIdx.x][threadIdx.y] = value[head_id * Lk * Dh + idx];
        
        // __syncthreads();

        // // Compute QKᵀ (dot product)
        // half dot_product = __float2half(0.0f);
        // for (int i = 0; i < BLOCK_SIZE; i++) {
        //     dot_product += shared_q[threadIdx.x][i] * shared_k[threadIdx.y][i];
        // }
        
        // float dot_f = __half2float(dot_product);
        // float sum_f = __half2float(sum_exp);   // assuming sum_exp is half

        // float softmax_f = expf(dot_f) / sum_f;
        // half softmax_out = __float2half(softmax_f);
        
        // Multiply with value to get final output
        out[0] = __float2half(42.0f);
    }
}

void launch_flash_attention_kernel(
    half* q,
    half* k,
    half* v,
    half* out,
    int Lq, int Lk, int H, int Dh
) {
    dim3 grid(1);
    dim3 block(1);
    flash_attention_kernel<<<grid, block>>>(q, k, v, out, Lq, Lk, H, Dh);
}

/*
    q (B, H, Lq, Dh)
    k (B, H, Lk, Dh)
    v (B, H, Lk, Dh)


*/
__global__ void attention_kernel(half* query, half* key, half* value, half* out, int Lq, int Lk, int H, int Dh) {
    // half* softmax_o;
    // cudaMalloc
    // for (int i = 0; i < B; i++)
    //     for (int j = 0; j < H; j++) {
    //         for (int lq = 0; lq < Lq; lq++)
    //             for (int lk = 0; lk < Lk; lk++)
    //                 for (int k = 0; k < Dh; k++)
                        

    //     }
}

# without any optimization
void launch_attention_kernel() {
    dim3 grid(1);
    dim3 block(1);
    attention_kernel<<<grid, block>>>(q, k, v, out, Lq, Lk, H, Dh);
}
