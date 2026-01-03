#pragma once
#include <cuda_fp16.h>  

__global__ void flash_attention_kernel(half* query, half* key, half* value, half* out, int Lq, int Lk, int H, int Dh);
void launch_flash_attention_kernel(half* q, half* k, half* v, half* out, int Lq, int Lk, int H, int Dh);