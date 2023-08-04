## llama2.zig

This is an implementation of the excellent [llama2.c](https://github.com/karpathy/llama2.c) by
Andrej Karpathy, translated into Zig.

My goal is to learn Zig while simultaneously gaining a better understanding of LLMs. Admittedly, I
don't have much expertise in either of these topics :)

If anyone takes the time to read through my code, I'd appreciate any feedback that can improve my
Zig programming skills. Admittedly, the code is a bit rough around the edges at the moment. My
initial focus was to produce output that is compatible with llama2.c.

The current code is based on
https://github.com/karpathy/llama2.c/blob/af8708d87bcda7fda97b93f6c135dd43ea78106c/run.c

I have attempted to stay true to the philosophy of the original. The only dependency is the Zig
`std` library. I have, however, divided it into several files for better clarity.

Some deviations from the original include:

- No OpenMP support; therefore, it runs only on one core
- SIMD optimization of the matmul function using `@Vector`
- No mmap support; the checkpoint file is instead fully loaded into the RAM
  - which I suspect explains the relatively good performance of LLaMA2 7B compared to the C
    implementation
- Utilization of slices instead of many-item pointers

## todos

- Currently, the temperature cannot be adjusted, the next token is deterministically selected via
  `argmax`
- The first transformer run (`pos=0`) takes disproportionately long (observed with LLaMA2 7B)

## performance

Testing performed on system Apple M1 Pro 32 GB

### build command

- Zig: `zig build run -Doptimize=ReleaseFast -- stories15M.bin 256`
- C: `make runfast`

### stories15M.bin

- Zig: **682 tokens/sec**
- C: 669 tokens/sec

### stories42M.bin

- Zig: **266 tokens/sec**
- C: 264 tokens/sec

### stories110M.bin

- Zig: **102 tokens/sec**
- C: 99 tokens/sec

### llama2_7b.bin

- Zig: **1 token/sec**
- C: slow...
