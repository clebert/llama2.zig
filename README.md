# llama2.zig

> Inference Llama 2 in pure Zig

<img src="logo.png" width="50%" height="50%">

This project is a port of Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) into
Zig, aimed at enhancing understanding of transformer models through clean, well-structured code.
Utilizing a multi-file approach and descriptive variable names, it relies exclusively on the Zig
standard library, without the need for external dependencies.

## Usage

Build and run `llama2-generator`:

```sh
zig build -Doptimize=ReleaseFast
```

```sh
./zig-out/bin/llama2-generator models/tinystories_15m --temperature 0 --worker_count 0
```

Output:

```
Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty.
Lily wanted to play with the ball, but it was too high up in the sky. She tried to jump and reach it, but she couldn't. Then, she had an idea. She would use a stick to knock the ball down.
Lily found a stick and tried to hit the ball. But the stick was too short. She tried again and again, but she couldn't reach it. She felt sad.
Suddenly, a kind man came by and saw Lily. He asked her what was wrong. Lily told him about the ball. The man smiled and said, "I have a useful idea!" He took out a long stick and used it to knock the ball down. Lily was so happy! She thanked the man and they played together in the sunshine.
```

## Run Llama 2 7B from Hugging Face

Install `git-lfs` and clone the [Llama 2 7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) model
from Hugging Face:

```sh
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
```

```sh
git clone https://huggingface.co/meta-llama/Llama-2-7b-hf
```

Install the necessary Python packages and convert the Hugging Face model:

```sh
pip3 install -r requirements.txt
```

```sh
python3 convert_hf_model.py /path/to/Llama-2-7b-hf models/llama2_7b_hf
```

Build and run `llama2-generator`:

```sh
zig build -Doptimize=ReleaseFast
```

```sh
./zig-out/bin/llama2-generator models/llama2_7b_hf \
  --prompt "Once Upon a Time" \
  --sequence_length 28 \
  --temperature 0
```

Output:

```
Once Upon a Time in Hollywood is a 2019 American comedy-drama film written and directed by Quentin Tarantino
```

## Run Llama 2 7B Chat from Hugging Face

Install `git-lfs` and clone the
[Llama 2 7B Chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) model from Hugging Face:

```sh
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
```

```sh
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
```

Install the necessary Python packages and convert the Hugging Face model:

```sh
pip3 install -r requirements.txt
```

```sh
python3 convert_hf_model.py /path/to/Llama-2-7b-chat-hf models/llama2_7b_chat_hf
```

Build and run `llama2-chat`:

```sh
zig build -Doptimize=ReleaseFast
```

```sh
./zig-out/bin/llama2-chat models/llama2_7b_chat_hf --temperature 0
```

Output:

```
Enter system prompt (optional):
User: Hello
Assistant: Hello! It's nice to meet you. Is there something I can help you with or would you like to chat?
User: ...
```

## Help

### llama2-generator

```
Usage: llama2-generator <model_path> [options]

Options:
  --help
  --prompt          <string> = ""
  --random_seed     <int>    = <milli_timestamp>
  --sequence_length <int>    = <max_sequence_length>
  --temperature     <float>  = 1.0
  --top_p           <float>  = 0.9
  --verbose
  --worker_count    <int>    = <cpu_count>
```

### llama2-chat

```
Usage: llama2-chat <model_path> [options]

Options:
  --help
  --random_seed     <int>    = <milli_timestamp>
  --sequence_length <int>    = <max_sequence_length>
  --system_prompt   <string> = ""
  --temperature     <float>  = 1.0
  --top_p           <float>  = 0.9
  --user_prompt     <string> = ""
  --worker_count    <int>    = <cpu_count>
```

## Papers

- Standard transformer architecture: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Llama 1: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- Llama 2: [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- Pre-normalization using RMSNorm:
  [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- SwiGLU activation function: [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- Swish activation function: [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)
- Rotary positional embeddings:
  [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- Grouped-query attention:
  [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245v1)
- Nucleus sampling: [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)

## Benchmark Results

The following benchmark results are categorized by CPU, Model, and Worker Count. The worker count
indicates the number of threads used for matrix-vector multiplications. Zero workers are quicker
than one as the latter leads to unnecessary overhead. Only with larger models, having workers
becomes beneficial. Prior to that, the performance gain doesn't surpass the overhead.

### TL;DR

The 15M model presents its fastest performance in single-threaded mode. For the 42M/110M models,
they both present their fastest performance on the M2 Pro with the use of 7 extra threads, and on
the M1 Pro with the use of 5 extra threads.

### Apple M2 Pro 8/4 32 GB

- Runs: 100
- Command:
  `./zig-out/bin/llama2-generator "$model" --temperature 0 --verbose --worker_count "$worker_count"`
- Zig Version: `0.12.0-dev.1261+bb0419599`
- Commit:
  [415247a0c09306bcfab9b491afa40179943b241c](https://github.com/clebert/llama2.zig/tree/415247a0c09306bcfab9b491afa40179943b241c)

#### models/tinystories_15m

| Worker Count | Avg (tok/s) | Min  | Max |
| ------------ | ----------- | ---- | --- |
| **0**        | **764**     | -23  | +30 |
| 1            | 623         | -26  | +15 |
| 2            | 645         | -12  | +14 |
| 3            | 671         | -16  | +17 |
| 4            | 634         | -12  | +14 |
| 5            | 669         | -14  | +17 |
| 6            | 662         | -45  | +20 |
| 7            | 627         | -33  | +24 |
| 8            | 597         | -13  | +11 |
| 9            | 567         | -18  | +14 |
| 10           | 538         | -11  | +20 |
| 11           | 505         | -140 | +15 |
| 12           | 484         | -7   | +14 |

#### models/tinystories_42m

| Worker Count | Avg (tok/s) | Min | Max |
| ------------ | ----------- | --- | --- |
| 0            | 288         | -3  | +4  |
| 1            | 246         | -4  | +6  |
| 2            | 268         | -4  | +6  |
| 3            | 284         | -7  | +8  |
| 4            | 293         | -21 | +14 |
| 5            | 306         | -4  | +5  |
| 6            | 331         | -4  | +5  |
| **7**        | **336**     | -5  | +7  |
| 8            | 320         | -3  | +6  |
| 9            | 306         | -4  | +6  |
| 10           | 296         | -4  | +4  |
| 11           | 283         | -54 | +6  |
| 12           | 273         | -51 | +5  |

#### models/tinystories_110m

| Worker Count | Avg (tok/s) | Min | Max |
| ------------ | ----------- | --- | --- |
| 0            | 106         | -2  | +1  |
| 1            | 96          | -1  | +1  |
| 2            | 108         | 0   | +1  |
| 3            | 110         | -1  | +1  |
| 4            | 116         | -1  | +4  |
| 5            | 124         | 0   | +2  |
| 6            | 139         | 0   | +2  |
| **7**        | **147**     | -1  | +2  |
| 8            | 144         | -3  | +2  |
| 9            | 138         | -1  | +4  |
| 10           | 134         | -1  | +3  |
| 11           | 130         | -11 | +1  |
| 12           | 127         | -11 | +8  |

### Apple M1 Pro 8/2 32 GB

- Runs: 100
- Command:
  `./zig-out/bin/llama2-generator "$model" --temperature 0 --verbose --worker_count "$worker_count"`
- Zig Version: `0.12.0-dev.1253+b798aaf49`
- Commit:
  [415247a0c09306bcfab9b491afa40179943b241c](https://github.com/clebert/llama2.zig/tree/415247a0c09306bcfab9b491afa40179943b241c)

#### models/tinystories_15m

| Worker Count | Avg (tok/s) | Min | Max |
| ------------ | ----------- | --- | --- |
| **0**        | **704**     | -29 | +25 |
| 1            | 596         | -28 | +16 |
| 2            | 647         | -28 | +26 |
| 3            | 627         | -14 | +17 |
| 4            | 607         | -44 | +15 |
| 5            | 568         | -21 | +15 |
| 6            | 555         | -62 | +14 |
| 7            | 542         | -32 | +16 |
| 8            | 502         | -60 | +22 |
| 9            | 487         | -33 | +16 |
| 10           | 477         | -47 | +20 |

#### models/tinystories_42m

| Worker Count | Avg (tok/s) | Min | Max |
| ------------ | ----------- | --- | --- |
| 0            | 269         | -7  | +5  |
| 1            | 235         | -14 | +3  |
| 2            | 262         | -13 | +5  |
| 3            | 262         | -8  | +3  |
| 4            | 284         | -10 | +5  |
| **5**        | **292**     | -8  | +6  |
| 6            | 261         | -7  | +7  |
| 7            | 259         | -5  | +10 |
| 8            | 259         | -11 | +15 |
| 9            | 264         | -9  | +6  |
| 10           | 259         | -11 | +6  |

#### models/tinystories_110m

| Worker Count | Avg (tok/s) | Min | Max |
| ------------ | ----------- | --- | --- |
| 0            | 99          | -1  | +2  |
| 1            | 91          | -3  | +1  |
| 2            | 102         | -3  | +19 |
| 3            | 103         | -1  | +4  |
| 4            | 119         | -2  | +1  |
| **5**        | **127**     | -5  | +2  |
| 6            | 123         | -1  | +1  |
| 7            | 124         | 0   | +2  |
| 8            | 124         | -2  | +4  |
| 9            | 123         | -3  | +4  |
| 10           | 121         | -3  | +3  |
