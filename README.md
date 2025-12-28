# flashattention_from_scratch

## Workload breakdown from GPT(day-level) 

### Phase 0 — Prep & design (½–1 day)

**Goal:** avoid dead ends

Tasks:

- Fix tensor layout: `[B, H, L, Dh]`
- Decide tile sizes (initial guess, not optimal)
- Decide comparison baseline (PyTorch SDPA / flash-attn)

⏱️ **0.5–1 day**

------

### Phase 1 — Reference correctness (1–2 days)

**Goal:** lock down math & numerics

Tasks:

- Naive PyTorch attention (with causal mask)
- Implement **online softmax in Python**
- Validate against `torch.nn.functional.scaled_dot_product_attention`

This saves you *days* later.

⏱️ **1–2 days**

------

### Phase 2 — First CUDA kernel (core milestone) (2–3 days)

**Goal:** fused forward kernel that works

Tasks:

- CUDA kernel with:
  - tiled Q×K
  - online softmax
  - accumulation into output
- No tensor cores yet
- FP32 accumulation OK initially

Milestone:

> “I have a single kernel, correct output, slower than PyTorch but fused.”

⏱️ **2–3 days**

This is the **hardest conceptual step**, but very educational.

------

### Phase 3 — Make it “Flash-like” (2–3 days)

**Goal:** real FlashAttention properties

Tasks:

- Switch to FP16/BF16
- Reduce global memory traffic
- Tune tile sizes
- Move more into registers
- Fix shared-memory layout issues

Milestone:

> “My kernel is clearly faster than naive attention.”

⏱️ **2–3 days**

------

### Phase 4 — PyTorch extension + Qwen integration (1–2 days)

**Goal:** real model, not toy code

Tasks:

- PyTorch C++/CUDA extension
- Python wrapper
- Patch Qwen attention forward to call your kernel
- Run real prompts through Qwen

⏱️ **1–2 days**

This is mostly engineering, not algorithmic pain.

------

### Phase 5 — Benchmarking & profiling (1–2 days)

**Goal:** credible performance comparison

Tasks:

- Microbench (attention only)
- End-to-end prefill timing in Qwen
- Nsight Compute screenshots
- Latency vs sequence length plots

⏱️ **1–2 days**
