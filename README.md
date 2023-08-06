## llama2.zig

> Inference Llama 2 in pure Zig

This is an implementation of the excellent [llama2.c](https://github.com/karpathy/llama2.c) by
Andrej Karpathy, translated into Zig.

My goal is to learn Zig while simultaneously gaining a better understanding of LLMs.

The current code is based on
https://github.com/karpathy/llama2.c/blob/d1a59a9ca8a07756ffdb7036bbfed83cc0179a3e/run.c

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
- The variable `v_len` in the `matmul` function is currently set to a static 32. This value defines
  the size of the SIMD vectors. A value below 8 or above 64 leads to significantly worse results in
  my tests. Perhaps a different value would be beneficial for Intel/AMD processors?

## performance

Testing performed on system Apple M1 Pro 32 GB

### run command

- Zig: `zig build run -Doptimize=ReleaseFast -- stories15M.bin -t 0.9 -n 256`
- C: `make runfast && ./run stories15M.bin -t 0.9 -p 0 -n 256`

### stories15M.bin

- Zig: 605 token/sec
- C: 668 token/sec

### stories42M.bin

- Zig: 246 token/sec
- C: 263 token/sec

### stories110M.bin

- Zig: 98 token/sec
- C: 100 token/sec

### llama2_7b.bin

- Zig: 2 token/sec
