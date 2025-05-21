#include <stdio.h>
#include <stdlib.h>

#include "utils.h"
#include "amm_dtype.h"
#include "ndarray.h"

void test_ndarray_creation() {
  float* storage1 = malloc(3 * 4 * 5 * sizeof(float));
  float* storage2 = malloc(3 * 4 * 5 * sizeof(float));

  int shape1[3] = {3, 4, 5};
  int shape2[3] = {3, 4, 5};
  amm_make_row_major_shape(3, shape1);
  NDArray* arr1 = amm_ndarray_alloc(amm_make_row_major_shape(3, shape1), storage1, AMM_DTYPE_F32);
  NDArray* arr2 = amm_ndarray_alloc(amm_make_column_major_shape(3, shape2), storage2, AMM_DTYPE_F32);
        
  printf("Passed: test_ndarray_creation\n");
}

int main(void) {
  test_ndarray_creation();
}
