#include "ndarray.h"
#include "utils.h"
#include "amm_dtype.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

NDArray* amm_ndarray_alloc(int ndim, int* shape, int* strides, void* storage, AMM_DType dtype) {
  struct NDArray *arr = malloc(sizeof *arr);
  if (!arr) {
    fprintf(stderr, "Failed to allocate memory for NDArray\n");
    return NULL;
  }
  arr->ndim = ndim;
  arr->shape = malloc(ndim * sizeof(int));
  memcpy(arr->shape, shape, ndim * sizeof(int));
  arr->strides = malloc(ndim * sizeof(int));
  memcpy(arr->strides, strides, ndim * sizeof(int));
  arr->storage = storage;
  arr->dtype = dtype;
  return arr;
}

void amm_ndarray_free(NDArray* arr) {
  if (arr) {
    free(arr->shape);
    free(arr->strides);
    free(arr);
  }
}
