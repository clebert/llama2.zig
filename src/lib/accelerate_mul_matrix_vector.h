#include <stdint.h>

void accelerateMulMatrixVector(
    const float *row_major_matrix,
    const float *input_vector,
    float *output_vector,
    int64_t m_rows,
    int64_t n_cols);
