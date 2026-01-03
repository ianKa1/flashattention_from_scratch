#include <cuda_runtime.h>
#include <iostream>
#include "flash_attn_cuda.h"

#define B 1
#define H 8
#define Lq 16
#define Dh 128
#define Lk 16

void init_tensor_random(half* tensor, int size) {
    half* temp = new half[size];
    for (int i = 0; i < size; ++i) {
        float rnd = static_cast<float>(rand()) / RAND_MAX;
        temp[i] = __float2half(rnd);
    }
    cudaMemcpy(tensor, temp, size * sizeof(half), cudaMemcpyHostToDevice);
    delete[] temp;
}

int main() {
    half* query;
    half* key;
    half* value;
    half* output;
    init_tensor_random(query, B * H * Lq * Dh);
    init_tensor_random(key, B * H * Lk * Dh);
    init_tensor_random(value, B * H * Lk * Dh);
    cudaMalloc((void**)&output, B * H * Lq * Dh * sizeof(half));
    cudaMemset(output, 0, B * H * Lq * Dh * sizeof(half));
    launch_flash_attention_kernel(query, key, value, output, Lq, Lk, H, Dh);

    half* output_host = new half[B * H * Lq * Dh];
    cudaMemcpy(output_host, output, B * H * Lq * Dh * sizeof(half), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 20; i++)
        printf("%f ", __half2float(output_host[i]));


    
    delete[] output_host;
    cudaFree(query);
    cudaFree(key);
    cudaFree(value);
    cudaFree(output);
    return 0;    
}