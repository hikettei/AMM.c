#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

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

  int valid_stride1[3] = {20, 5, 1};
  int valid_stride2[3] = {1, 3, 12};
  
  amm_assert(arr1->shape->is_contiguous == true, "arr1 should be contiguous");
  amm_assert(arr2->shape->is_contiguous == true, "arr2 should be contiguous");
  for (int i = 0; i < 3; i++ ) {
    amm_assert(amm_ndarray_stride_of(arr1, i) == valid_stride1[i], "Invalid stride for arr1 at dimension %d: expected %d, got %d. ", i, valid_stride1[i], amm_ndarray_stride_of(arr1, i));
    amm_assert(amm_ndarray_stride_of(arr2, i) == valid_stride2[i], "Invalid stride for arr2 at dimension %d: expected %d, got %d. ", i, valid_stride2[i], amm_ndarray_stride_of(arr2, i));
  }
  // -1 Accessing
  amm_assert(amm_ndarray_size_of(arr1, -1) == 5, "Invalid size for arr1 at dimension -1: expected 5, got %d. ", amm_ndarray_size_of(arr1, -1));
  amm_assert(amm_ndarray_stride_of(arr1, -1) == 1, "Invalid stride for arr1 at dimension -1: expected 1, got %d. ", amm_ndarray_stride_of(arr1, -1));
  print_ndarray(arr1);
  print_ndarray(arr2);
  amm_ndarray_free(arr1);
  amm_ndarray_free(arr2);
  printf("Passed: test_ndarray_creation\n");
}

void test_ndarray_arange_and_contiguous_elwise() {
  NDArray* arr1 = amm_ndarray_zeros(amm_make_row_major_shape(2, (int[]){10, 10}), AMM_DTYPE_F32);
  NDArray* arr2 = amm_ndarray_zeros(amm_make_row_major_shape(2, (int[]){10, 10}), AMM_DTYPE_F32);
  arr1 = amm_ndarray_index_components(arr1);
  arr2 = amm_ndarray_index_components(arr2);
  arr1 = amm_ndarray_add(arr1, arr2);
  print_ndarray(arr1);
  print_ndarray(arr2);
  amm_ndarray_free(arr1);
  amm_ndarray_free(arr2);
  printf("Passed: test_ndarray_arange_and_contiguous_elwise\n");
}

void test_ndarray_reshape() {
  NDArray* arr1 = amm_ndarray_zeros(amm_make_row_major_shape(2, (int[]){10, 10}), AMM_DTYPE_F32);
  NDArray* arr2 = amm_ndarray_reshape(arr1, amm_make_row_major_shape(3, (int[]){1, 1, 100}));
  print_ndarray(arr1);
  print_ndarray(arr2);
  int valid_shape[3] = {1, 1, 100};
  int valid_stride[3] = {100, 100, 1};
  for (int i = 0; i < 3; i++) {
    amm_assert(amm_ndarray_size_of(arr2, i) == valid_shape[i], "Invalid size for arr2");
    amm_assert(amm_ndarray_stride_of(arr2, i) == valid_stride[i], "Invalid stride for arr2");
  }
  printf("Passed: test_ndarray_reshape\n");
}

int main(void) {
  test_ndarray_creation();
  test_ndarray_arange_and_contiguous_elwise();
  test_ndarray_reshape();
}
