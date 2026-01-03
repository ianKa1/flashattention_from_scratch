import torch
import flash_attn_ext  # The name of your extension
import torch.nn.functional as F

# Create tensor inputs (on GPU)
query = torch.randn(128, 8, 128, 64, device='cuda', dtype=torch.float16)
key = torch.randn(128, 8, 128, 64, device='cuda', dtype=torch.float16)
value = torch.randn(128, 8, 128, 64, device='cuda', dtype=torch.float16)
out = torch.zeros_like(query)

# Call the kernel (wrapped in your PyTorch extension)
out = flash_attn_ext.flash_attn_forward(query, key, value)
out_ref = F.scaled_dot_product_attention(query, key, value)

print(out)