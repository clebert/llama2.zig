## llama2.zig

> Inference Llama 2 in pure Zig

This is a Zig port of [llama2.c](https://github.com/karpathy/llama2.c).

My goal is to learn Zig while simultaneously gaining a better understanding of LLMs.

The current code is based on
https://github.com/karpathy/llama2.c/blob/bd182289c596fa6059eb7b3b7c8ccd04b5c90fc3/run.c

I have attempted to stay true to the philosophy of the original. The only dependency is the Zig
`std` library. I have, however, divided it into several files for better clarity.

Some deviations from the original include:

- No OpenMP support
- SIMD optimization of the matmul function using `@Vector`
- Utilization of slices instead of many-item pointers
- For models of 4096+ dimensions, thread pools are utilized to parallelize independent matrix
  multiplications

## Papers

- Standard transformer architecture: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Llama 1: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- Llama 2: [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- Pre-normalization using RMSNorm: [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- SwiGLU activation function: [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- Rotary positional embeddings: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- Grouped-query attention: [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245v1)
