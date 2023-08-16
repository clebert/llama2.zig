## llama2.zig

> Inference Llama 2 in pure Zig

This is a Zig port of [llama2.c](https://github.com/karpathy/llama2.c).

My goal is to learn Zig while simultaneously gaining a better understanding of LLMs.

The current code is based on
https://github.com/karpathy/llama2.c/blob/ca67253f28f95b11a8d3b76a3058eccd70c2b471/run.c

I have attempted to stay true to the philosophy of the original. The only dependency is the Zig
`std` library. I have, however, divided it into several files for better clarity.

Some deviations from the original include:

- No OpenMP support
- SIMD optimization of the matmul function using `@Vector`
- Utilization of slices instead of many-item pointers
- For models of 4096+ dimensions, thread pools are utilized to parallelize independent matrix
  multiplications
