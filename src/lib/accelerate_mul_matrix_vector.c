#define ACCELERATE_NEW_LAPACK
#define ACCELERATE_LAPACK_ILP64

#include <Accelerate/Accelerate.h>
#include "accelerate_mul_matrix_vector.h"

void accelerateMulMatrixVector(
    const float *row_major_matrix,
    const float *input_vector,
    float *output_vector,
    int64_t m_rows,
    int64_t n_cols)
{
  // https://developer.apple.com/documentation/accelerate/1513065-cblas_sgemv
  cblas_sgemv(
      CblasRowMajor,
      CblasNoTrans,
      m_rows,
      n_cols,
      1,
      row_major_matrix,
      n_cols,
      input_vector,
      1,
      0,
      output_vector,
      1);
}
