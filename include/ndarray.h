#include "amm_dtype.h"

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
