#include "ndarray.h"
#include "utils.h"
#include "amm_dtype.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdbool.h>

/*
  ShapeTracker Initialization
*/
__amm_give Shape* amm_make_strided_shape(int nrank, const int* shape, const int* stride)
{
  if (nrank <= 0) {
    fprintf(stderr, "amm_make_strided_shape: invalid nrank=%d\n", nrank);
    return NULL;
  }
  Shape* s = malloc(sizeof *s);
  if (!s) {
    fprintf(stderr, "amm_make_strided_shape: failed to alloc Shape\n");
    return NULL;
  }
  s->nrank         = nrank;
  s->is_contiguous = true;
  s->axes          = malloc(nrank * sizeof *s->axes);
  if (!s->axes) {
    fprintf(stderr, "amm_make_strided_shape: failed to alloc axes array\n");
    free(s);
    return NULL;
  }
  for (int i = 0; i < nrank; ++i) {
    Axis* ax = malloc(sizeof *ax);
    if (!ax) {
      fprintf(stderr, "amm_make_strided_shape: failed to alloc Axis[%d]\n", i);
      for (int j = 0; j < i; ++j) free(s->axes[j]);
      free(s->axes);
      free(s);
      return NULL;
    }
    ax->size              = shape[i];
    ax->offset            = 0;
    ax->stride            = stride[i];
    ax->random_access_idx = NULL;
    s->axes[i] = ax;
    // TODO: Verify the contiguous of stride here?
  }
  return s;
}

__amm_give Shape* amm_make_column_major_shape(int nrank, int* shape) {
  int* stride = malloc(sizeof(int) * nrank);
  if (!stride) {
    fprintf(stderr, "Failed to allocate memory for stride array\n");
    return NULL;
  }
  stride[0] = 1;
  for (int i = 1; i < nrank; ++i) {
    stride[i] = stride[i-1] * shape[i-1];
  }
  Shape* s = amm_make_strided_shape(nrank, shape, stride);
  free(stride);
  return s;
}

__amm_give Shape* amm_make_row_major_shape(int nrank, int* shape) {
  int* stride = malloc(sizeof(int) * nrank);
  if (!stride) {
    fprintf(stderr, "Failed to allocate memory for stride array\n");
    return NULL;
  }
  stride[nrank - 1] = 1;
  for (int i = nrank - 2; i >= 0; --i) {
    stride[i] = stride[i+1] * shape[i+1];
  }
  Shape* s = amm_make_strided_shape(nrank, shape, stride);
  free(stride);
  return s;
}

void amm_free_axis(Axis* axis) {
  if (!axis) return;
  if (axis->random_access_idx != NULL) free(axis->random_access_idx);
  free(axis);
}

void amm_free_shape(__amm_take Shape *s) {
  if (!s) return;
  for (int i = 0; i < s->nrank; ++i) amm_free_axis(s->axes[i]);
  free(s->axes);
  free(s);
}

__amm_give NDArray* amm_ndarray_alloc(Shape* shape, void* storage, AMM_DType dtype) {
  if (!shape) {
    fprintf(stderr, "amm_ndarray_alloc: shape is NULL\n");
    return NULL;
  }
  if (!storage) {
    fprintf(stderr, "amm_ndarray_alloc: storage is NULL\n");
    return NULL;
  }
  NDArray* arr = malloc(sizeof *arr);
  if (!arr) {
    fprintf(stderr, "amm_ndarray_alloc: failed to alloc NDArray\n");
    return NULL;
  }
  arr->shape = shape;
  arr->storage = storage;
  arr->dtype = dtype;
  return arr;
}

void amm_ndarray_free(__amm_take NDArray* arr) {
  if (arr) {
    amm_free_shape(arr->shape);
    free(arr->storage);
    free(arr);
  }
}
// ~~~ Accessors ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
int amm_ndarray_rank(__amm_keep NDArray* arr) {
  return arr ? arr->shape->nrank : 0;
}

int amm_ndarray_size_of(__amm_keep NDArray* arr, int dim) {
  if (!arr) return 0;
  if (dim < 0 || dim >= arr->shape->nrank) return 0;
  return arr->shape->axes[dim]->size;
}

int amm_ndarray_stride_of(__amm_keep NDArray* arr, int dim) {
  if (!arr) return 0;
  if (dim < 0 || dim >= arr->shape->nrank) return 0;
  return arr->shape->axes[dim]->stride;
}
// ~~~ Memory Allocations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
