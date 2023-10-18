# llama2.zig

> Inference Llama 2 in pure Zig

This project started as a Zig port of [llama2.c](https://github.com/karpathy/llama2.c).

## Usage

```sh
zig build -Doptimize=ReleaseFast run-generator -- models/tinystories_15m --temperature 0 --verbose
```

## Run Llama 2 from Hugging Face

```sh
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-7b-hf
```

```sh
pip3 install -r requirements.txt
```

```sh
python3 convert_hf_model.py /path/to/Llama-2-7b-hf models/llama2_7b_hf
```

```sh
zig build -Doptimize=ReleaseFast run-generator -- models/llama2_7b_hf --prompt "Once Upon a Time"
```

## Help

### llama2-generator

```
Usage: llama2-generator <model_path> [options]

Options:
  --temperature   <float>  = 1.0
  --top_p         <float>  = 0.9
  --random_seed   <int>    = <milli_timestamp>
  --n_steps       <int>    = <max_sequence_length>
  --prompt        <string> = ""
  --verbose
  --help
```

### llama2-chat

```
Usage: llama2-chat <model_path> [options]

Options:
  --temperature   <float>  = 1.0
  --top_p         <float>  = 0.9
  --random_seed   <int>    = <milli_timestamp>
  --n_steps       <int>    = <max_sequence_length>
  --prompt        <string> = ""
  --system_prompt <string> = ""
  --help
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
