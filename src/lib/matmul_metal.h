#include <iostream>

#ifdef __cplusplus
extern "C"
{
#endif

  void matmulMetal(float *result, const float *a, const float *b, size_t result_len, size_t a_len);

#ifdef __cplusplus
}
#endif
