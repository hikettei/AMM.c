#include "amm_dtype.h"
#pragma once
/*
  NDArray is a wrapper for *storage with shape and strides.
*/

typedef struct NDArray NDArray;
struct NDArray {
  int ndim;
  int *shape;
  int *strides;
  void* storage;
  AMM_DType dtype;
};

NDArray* amm_ndarray_alloc(int ndim, int* shape, int* strides, void* storage, AMM_DType dtype);
void amm_ndarray_free(NDArray* arr);

// TODO: Naming Convention inspired from ISL
// __amm_give__ __amm_take__ __amm_keep__
#define __amm_give
#define __amm_take
#define __amm_keep

__amm_give NDArray* amm_ndarray_give_randn();
