#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  void matvecmulMetal(const float *row_major_matrix, const float *input_vector, float *output_vector, u_int64_t m_rows, u_int64_t n_cols);

#ifdef __cplusplus
}
#endif
