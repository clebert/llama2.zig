## llama2.zig

> Inference Llama 2 in pure Zig

This is an implementation of the excellent [llama2.c](https://github.com/karpathy/llama2.c) by
Andrej Karpathy, translated into Zig.

My goal is to learn Zig while simultaneously gaining a better understanding of LLMs.

The current code is based on
https://github.com/karpathy/llama2.c/blob/af8708d87bcda7fda97b93f6c135dd43ea78106c/run.c

I have attempted to stay true to the philosophy of the original. The only dependency is the Zig
`std` library. I have, however, divided it into several files for better clarity.

Some deviations from the original include:

- No OpenMP support
- SIMD optimization of the matmul function using `@Vector`
- No mmap support; the checkpoint file is instead fully loaded into the RAM
- Utilization of slices instead of many-item pointers
- For models with more than 4096 dimensions, I divide two sentences of matrix multiplications across
  multiple threads

## todos

- The first transformer run (`pos=0`) takes disproportionately long (observed with Llama 2 7B)

## performance

Testing performed on system Apple M1 Pro 32 GB

### build command

- Zig: `zig build run -Doptimize=ReleaseFast -- stories15M.bin 0.9 256`
- C: `make runfast && ./run stories15M.bin 0.9 256`

### llama2_7b.bin

- Zig: **2 tokens/sec** 🎉
- C: slow...

### stories15M.bin

- Zig: 564 tokens/sec
- C: 620 tokens/sec

### stories42M.bin

- Zig: 235 tokens/sec
- C: 246 tokens/sec

### stories110M.bin

- Zig: 92 tokens/sec
- C: 95 tokens/sec
