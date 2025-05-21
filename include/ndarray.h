#include "amm_dtype.h"
#include <stdbool.h>

#pragma once
#ifndef __amm_give
#define __amm_give 
#endif
#ifndef __amm_take
#define __amm_take
#endif
#ifndef __amm_keep
#define __amm_keep
#endif
/*
  NDArray is a wrapper for *storage with shape and strides.
*/
typedef struct Axis Axis;
struct Axis {
  int size;
  int offset;
  int stride;
  void* random_access_idx; // If random_access_idx was set to int* pointer, ndarray will read the corresponding index by random_access_idx[0], ... random_access_idx[size-1]
};

typedef struct Shape Shape;
struct Shape {
  int nrank;
  Axis** axes;
  bool is_contiguous;
};

typedef struct NDArray NDArray;
struct NDArray {
  Shape* shape;
  void* storage;
  AMM_DType dtype;
};
/*
  Shaping
*/
// Allocator
__amm_give Shape* amm_make_strided_shape(int nrank, const int* shape, const int* stride);
__amm_give Shape* amm_make_column_major_shape(int nrank, int* shape);
__amm_give Shape* amm_make_row_major_shape(int nrank, int* shape);
// Freer
void amm_free_axis(Axis* axis);
void amm_free_shape(__amm_take Shape* s);
/*
  Allocation
*/
__amm_give NDArray* amm_ndarray_alloc(Shape* shape, void* storage, AMM_DType dtype);
void amm_ndarray_free(__amm_take NDArray* arr);
/*
  Accessors
*/
int amm_ndarray_rank(__amm_keep NDArray* arr);
int amm_ndarray_size_of(__amm_keep NDArray* arr, int dim);
int amm_ndarray_stride_of(__amm_keep NDArray* arr, int dim);
// ~~~ Initializers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
__amm_give NDArray* amm_ndarray_randn();
__amm_give NDArray* amm_ndarray_zeros();
// ~~~ Movements ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
__amm_keep NDArray* amm_ndarray_reshape(__amm_keep NDArray* arr);

