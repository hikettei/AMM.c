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

void test_ndarray_permute() {
  NDArray* arr1 = amm_ndarray_zeros(amm_make_row_major_shape(2, (int[]){10, 10}), AMM_DTYPE_F32);
  NDArray* arr2 = amm_ndarray_zeros(amm_make_row_major_shape(2, (int[]){10, 10}), AMM_DTYPE_F32);
  arr1 = amm_ndarray_index_components(arr1);
  arr2 = amm_ndarray_index_components(arr2);
  arr1 = amm_ndarray_permute(arr1, 1, 0);
  print_ndarray(arr1);
  print_ndarray(arr2);
  for (int i=0; i<10; i++)
    for (int j=0; j<10; j++)
      amm_assert(amm_ndarray_aref(float, arr1, i, j) == amm_ndarray_aref(float, arr2, j, i), "Invalid value at (%d, %d): expected %f", i, j, amm_ndarray_aref(float, arr1, j, i));
  amm_ndarray_free(arr1);
  amm_ndarray_free(arr2);
  printf("Passed: test_ndarray_permute\n");
}

void test_ndarray_expand() {
  NDArray* arr1 = amm_ndarray_zeros(amm_make_row_major_shape(2, (int[]){1, 1}), AMM_DTYPE_F32);
  NDArray* arr2 = amm_ndarray_zeros(amm_make_row_major_shape(2, (int[]){10, 10}), AMM_DTYPE_F32);
  amm_ndarray_apply_unary(float, x[x_i] = 1.0f, arr2);
  for (int i=0; i<100; i++) amm_assert(((float*)arr2->storage)[i] == 1.0f, "Invalid value at (%d): expected %f", i, ((float*)arr2->storage)[i]);
  amm_ndarray_expand(arr1, (int[]){10, 10});
  amm_ndarray_apply_binary(float, float, out[out_i] = out[out_i] + x[x_i], arr1, arr2);
  amm_assert(amm_ndarray_aref(float, arr1, 0, 0) == 100.0f, "Invalid value at (0, 0): expected 100.0");
  amm_ndarray_free(arr1); amm_ndarray_free(arr2);
  printf("Passed: test_ndarray_expand\n");
}

void test_ndarray_indexing_view() {
  NDArray* arr1 = amm_ndarray_zeros(amm_make_row_major_shape(2, (int[]){10, 10}), AMM_DTYPE_F32);
  arr1 = amm_ndarray_index_components(arr1);
  print_ndarray(arr1);
  amm_ndarray_view_index(arr1, 1, 3, (int[]){1, 3, 2});
  int valid_shape[3] = {1, 3, 2};
  for (int i=0; i<10; i++)
    for (int j=0; j<3; j++)
      amm_assert(amm_ndarray_aref(float, arr1, i, j) == i * 10 + valid_shape[j], "Invalid value at (%d, %d)", i, j);
  amm_ndarray_view_index(arr1, 0, 3, (int[]){1, 3, 2});
  for (int i=0; i<3; i++)
    for (int j=0; j<3; j++)
      amm_assert(amm_ndarray_aref(float, arr1, i, j) == valid_shape[i] * 10 + valid_shape[j], "Invalid value at (%d, %d)", i, j);
  amm_ndarray_free(arr1);
  printf("Passed: test_ndarray_indexing_view\n");
}

void test_ndarray_slice_1() {
  // Case1: offset=0, but size was changed w/ all dims.
  NDArray* arr = amm_ndarray_zeros(amm_make_row_major_shape(3, (int[]){5, 5, 5}), AMM_DTYPE_F32);
  arr = amm_ndarray_index_components(arr);
  print_ndarray(arr);
  
  amm_ndarray_slice(arr, 0, 0, 2, 1);
  amm_assert(amm_ndarray_size_of(arr, 0) == 2 && amm_ndarray_size_of(arr, 1) == 5 && amm_ndarray_size_of(arr, 2) == 5, "Invalid size for arr");
  for (int i=0; i<2; i++)
    for (int j=0; j<5; j++)
      for (int k=0; k<5; k++)
        amm_assert(amm_ndarray_aref(float, arr, i, j, k) == i * 5 * 5 + j * 5 + k, "Invalid value at (%d, %d, %d)", i, j, k);
  
  amm_ndarray_slice(arr, 1, 0, 2, 1);
  amm_assert(amm_ndarray_size_of(arr, 0) == 2 && amm_ndarray_size_of(arr, 1) == 2 && amm_ndarray_size_of(arr, 2) == 5, "Invalid size for arr");
  for (int i=0; i<2; i++)
    for (int j=0; j<2; j++)
      for (int k=0; k<5; k++)
        amm_assert(amm_ndarray_aref(float, arr, i, j, k) == i * 5 * 5 + j * 5 + k, "Invalid value at (%d, %d, %d)", i, j, k);
  
  amm_ndarray_slice(arr, 2, 0, 2, 1);
  amm_assert(amm_ndarray_size_of(arr, 0) == 2 && amm_ndarray_size_of(arr, 1) == 2 && amm_ndarray_size_of(arr, 2) == 2, "Invalid size for arr");
  for (int i=0; i<2; i++)
    for (int j=0; j<2; j++)
      for (int k=0; k<2; k++)
        amm_assert(amm_ndarray_aref(float, arr, i, j, k) == i * 5 * 5 + j * 5 + k, "Invalid value at (%d, %d, %d)", i, j, k);
  
  print_ndarray(arr);
  amm_ndarray_free(arr);
  printf("Passed: test_ndarray_slice_1\n");
}

void test_ndarray_slice_2() {
  // Case2: offset is set to > 0
  NDArray* arr = amm_ndarray_zeros(amm_make_row_major_shape(3, (int[]){5, 5, 5}), AMM_DTYPE_F32);
  arr = amm_ndarray_index_components(arr);
  print_ndarray(arr);
  
  amm_ndarray_slice(arr, 0, 2, 4, 1);
  amm_assert(amm_ndarray_size_of(arr, 0) == 2 && amm_ndarray_size_of(arr, 1) == 5 && amm_ndarray_size_of(arr, 2) == 5, "Invalid size for arr");
  for (int i=0; i<2; i++)
    for (int j=0; j<5; j++)
      for (int k=0; k<5; k++)
        amm_assert(amm_ndarray_aref(float, arr, i, j, k) == (i+2) * 5 * 5 + j * 5 + k, "Invalid value at (%d, %d, %d)", i, j, k);
  
  amm_ndarray_slice(arr, 1, 2, 4, 1);
  amm_assert(amm_ndarray_size_of(arr, 0) == 2 && amm_ndarray_size_of(arr, 1) == 2 && amm_ndarray_size_of(arr, 2) == 5, "Invalid size for arr");
  for (int i=0; i<2; i++)
    for (int j=0; j<2; j++)
      for (int k=0; k<5; k++)
        amm_assert(amm_ndarray_aref(float, arr, i, j, k) == (i+2) * 5 * 5 + (j+2) * 5 + k, "Invalid value at (%d, %d, %d)", i, j, k);
  
  amm_ndarray_slice(arr, 2, 2, 4, 1);
  amm_assert(amm_ndarray_size_of(arr, 0) == 2 && amm_ndarray_size_of(arr, 1) == 2 && amm_ndarray_size_of(arr, 2) == 2, "Invalid size for arr");
  for (int i=0; i<2; i++)
    for (int j=0; j<2; j++)
      for (int k=0; k<2; k++)
        amm_assert(amm_ndarray_aref(float, arr, i, j, k) == (i+2) * 5 * 5 + (j+2) * 5 + (k+2), "Invalid value at (%d, %d, %d)", i, j, k);
  
  print_ndarray(arr);
  amm_ndarray_free(arr);
  printf("Passed: test_ndarray_slice_2\n");
}


int main(void) {
  test_ndarray_creation();
  test_ndarray_arange_and_contiguous_elwise();
  test_ndarray_reshape();
  test_ndarray_permute();
  test_ndarray_expand();
  test_ndarray_indexing_view();
  test_ndarray_slice_1();
  test_ndarray_slice_2();
}
