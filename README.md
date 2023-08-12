## llama2.zig

> Inference Llama 2 in pure Zig

This is an implementation of the excellent [llama2.c](https://github.com/karpathy/llama2.c) by
Andrej Karpathy, translated into Zig.

My goal is to learn Zig while simultaneously gaining a better understanding of LLMs.

The current code is based on
https://github.com/karpathy/llama2.c/blob/3c3b19b14c3d5fbfbe15ad4ff3ff0cc9cb510595/run.c

I have attempted to stay true to the philosophy of the original. The only dependency is the Zig
`std` library. I have, however, divided it into several files for better clarity.

Some deviations from the original include:

- No OpenMP support
- SIMD optimization of the matmul function using `@Vector`
- No mmap support; the checkpoint file is instead fully loaded into the RAM
- Utilization of slices instead of many-item pointers
- For models of 4096+ dimensions, I divide two matrix multiplication sets across threads

## todos

- The first transformer run (`pos=0`) takes disproportionately long (observed with Llama 2 7B)
- Identification of further opportunities for multithreading

## performance

Testing performed on system Apple M1 Pro 32 GB

### run command

- Zig: `zig build -Doptimize=ReleaseFast && ./zig-out/bin/llama2 stories15M.bin -t 0`
- C: `make runfast && ./run stories15M.bin -t 0`

### stories15M.bin

- Zig: 673 token/sec
- C: 692 token/sec

### stories42M.bin

- Zig: 266 token/sec
- C: 266 token/sec

### stories110M.bin

- Zig: 102 token/sec
- C: 99 token/sec

### llama2_7b.bin

- Zig: 2 token/sec
