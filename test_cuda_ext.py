import torch
import flash_attn_ext  # This is your CUDA extension

def my_flash_attention_forward(query, key, value):
    # Ensure query, key, value are on the GPU
    assert query.is_cuda
    assert key.is_cuda
    assert value.is_cuda

    # Call the custom CUDA kernel function
    out = flash_attn_ext.forward(query, key, value)
    
    return out

print("hello?")
# Test with dummy data
B, H, L, Dh = 1, 8, 16, 128
query = torch.randn(B, H, L, Dh, device="cuda")
key = torch.randn(B, H, L, Dh, device="cuda")
value = torch.randn(B, H, L, Dh, device="cuda")

output = my_flash_attention_forward(query, key, value)
print(output.shape)