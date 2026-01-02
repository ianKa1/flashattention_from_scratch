#include <cuda_fp16.h>  // Required for half precision (FP16)
#include <torch/extension.h>

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
        out[head_id * Lq + idx] = __float2half(42.0f);
    }
}

void flash_attn_forward_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor &output
) {
    TORCH_CHECK(query.is_cuda(), "query must be CUDA");
    TORCH_CHECK(query.scalar_type() == at::kHalf, "query must be fp16");

    int B  = query.size(0);
    int H  = query.size(1);
    int Lq = query.size(2);
    int Dh = query.size(3);
    int Lk = key.size(2);

    // ---------------------------------------------
    // GRID & BLOCK CONFIGURATION
    // ---------------------------------------------
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 blocks(
        (Lq + BLOCK_SIZE - 1) / BLOCK_SIZE, // tiles over query length
        H,                                  // heads
        B                                   // batch
    );

    flash_attention_kernel<<<blocks, threads>>>(
        reinterpret_cast<half*>(query.data_ptr<at::Half>()),
        reinterpret_cast<half*>(key.data_ptr<at::Half>()),
        reinterpret_cast<half*>(value.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        Lq, Lk, H, Dh
    );

}
