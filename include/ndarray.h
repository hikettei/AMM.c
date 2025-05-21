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
void amm_axis_free(Axis* axis);
void amm_shape_free(__amm_take Shape* s);
/*
  Allocation
*/
__amm_give NDArray* amm_ndarray_alloc(Shape* shape, void* storage, AMM_DType dtype);
void amm_ndarray_free(__amm_take NDArray* arr);
/*
  Accessors
*/
static inline int amm_ndarray_rank(__amm_keep const NDArray* arr);
int amm_ndarray_size_of(__amm_keep NDArray* arr, int dim);
int amm_ndarray_stride_of(__amm_keep NDArray* arr, int dim);
bool amm_ndarray_is_contiguous(__amm_keep NDArray* arr);
// ~~~ Initializers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// __amm_give NDArray* amm_ndarray_randn();
// __amm_give NDArray* amm_ndarray_zeros();
// TODO: __amm_give NDArray* amm_ndarray_copy(__amm_keep NDArray* arr);
// ~~~ Movements ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// TODO: The only operation here is apply_map, (this can implement even matmul, im2col, which is enough for our goal)
__amm_keep NDArray* amm_ndarray_reshape(__amm_keep NDArray* arr, Shape* new_shape);
__amm_keep NDArray* amm_ndarray_permute(__amm_keep NDArray* arr, const int* perm);
__amm_keep NDArray* amm_ndarray_view(__amm_keep NDArray* arr, int* shape);
__amm_keep NDArray* amm_ndarray_expand(__amm_keep NDArray* arr, int* expand);

