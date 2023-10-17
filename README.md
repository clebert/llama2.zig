## llama2.zig

> Inference Llama 2 in pure Zig

This is a Zig port of [llama2.c](https://github.com/karpathy/llama2.c).

The current code is based on:
https://github.com/karpathy/llama2.c/blob/7325bab657406c427e7c1ca6575bace9a5982744/run.c

I have significantly diverged from the original in terms of architecture and implementation.
However, my goal is to continue porting the improvements and new features of Andrej's C version into
this codebase. At present, my Zig port produces the same output as the C version. I ensure this
through the following linked [tests](./test.sh).

## Installation

```sh
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone git@github.com:clebert/llama2.zig.git
```

## Usage

```sh
zig build -Doptimize=ReleaseFast run -- models/tinystories_15m -t 0 --verbose
```

## Papers

- Standard transformer architecture: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Llama 1: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- Llama 2: [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- Pre-normalization using RMSNorm: [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- SwiGLU activation function: [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- Swish activation function: [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
- Rotary positional embeddings: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- Grouped-query attention: [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245v1)
- Nucleus sampling: [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)

## Links

- [Finding the top-p elements as used in Nucleus Sampling](https://blog.virtual-void.net/2023/08/29/calculating-top-p/)
