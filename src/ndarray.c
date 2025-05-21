#include "ndarray.h"
#include "utils.h"
#ifndef AMM_UTILS_H
#error "ndarray.c: either of AMM_C_GCC_MODE or AMM_C_BLOCK_MODE must be defined by loading utils.h"
#endif
#include "amm_dtype.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdbool.h>
#include <math.h>
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

void amm_axis_free(Axis* axis) {
  if (!axis) return;
  if (axis->random_access_idx != NULL) free(axis->random_access_idx);
  free(axis);
}

void amm_shape_free(__amm_take Shape *s) {
  if (!s) return;
  for (int i = 0; i < s->nrank; ++i) amm_axis_free(s->axes[i]);
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
    amm_shape_free(arr->shape);
    free(arr->storage);
    free(arr);
  }
}
// ~~~ Accessors ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
int amm_ndarray_rank(__amm_keep const NDArray* arr) {
  return arr ? arr->shape->nrank : 0;
}

int amm_ndarray_size_of(__amm_keep const NDArray* arr, int dim) {
  if (!arr) return 0;
  amm_assert(arr->shape->nrank > 0, "Invalid shape for size_of");
  if (dim < 0) {
    amm_assert(arr->shape->nrank + dim >= 0, "Invalid dimension %d for size_of", dim);
    return amm_ndarray_size_of(arr, arr->shape->nrank + dim);
  }
  return arr->shape->axes[dim]->size;
}

int amm_ndarray_stride_of(__amm_keep const NDArray* arr, int dim) {
  if (!arr) return 0;
  amm_assert(arr->shape->nrank > 0, "Invalid shape for stride_of");
  if (dim < 0) {
    amm_assert(arr->shape->nrank + dim >= 0, "Invalid dimension %d for stride_of", dim);
    return amm_ndarray_stride_of(arr, arr->shape->nrank + dim);
  }
  return arr->shape->axes[dim]->stride;
}

bool amm_ndarray_is_contiguous(__amm_keep const NDArray* arr) {
  if (!arr) return false;
  return arr->shape->is_contiguous;
}

int amm_ndarray_total_size(__amm_keep const NDArray* arr) {
  if (!arr) return 0;
  int total_size = 1;
  for (int i = 0; i < amm_ndarray_rank(arr); i++) {
    total_size *= amm_ndarray_size_of(arr, i);
  }
  return total_size;
}
// ~~ Movements ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
__amm_keep NDArray* amm_ndarray_reshape(__amm_take NDArray* arr, Shape* new_shape) {
  // Reshape gives a new shape and new stride without changing the stride
  if (!arr) return 0;
  amm_assert(amm_ndarray_is_contiguous(arr), "Reshape only works for contiguous arrays");
  amm_shape_free(arr->shape); // Replaces the shape
  arr->shape = new_shape;
  return arr;
}

__amm_keep NDArray* amm_ndarray_permute(__amm_take NDArray* arr,
                                        const int* perm) {
  if (!arr) return NULL;
  int nrank = amm_ndarray_rank(arr);
  Axis** old_axes = arr->shape->axes;
  Axis** new_axes = malloc(nrank * sizeof *new_axes);
  if (!new_axes) {
    fprintf(stderr, "amm_ndarray_permute: failed to alloc new_axes\n");
    return NULL;
  }
  for (int i = 0; i < nrank; ++i) {
    int p = perm[i];
    amm_assert(p >= 0 && p < nrank, "amm_ndarray_permute: invalid perm[%d]=%d", i, p);
    new_axes[i] = old_axes[p];
  }
  // replace axes array and free old
  arr->shape->axes = new_axes;
  free(old_axes);
  return arr;
}
// ~~ Apply ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


// ~~ Operations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

__amm_keep NDArray* amm_ndarray_sin(__amm_take NDArray* arr) {
  switch (arr->dtype) { // TODO: Make Switch Macro
  case AMM_DTYPE_F32:
    amm_ndarray_apply_f_unary(float, sinf, arr);
    break;
  case AMM_DTYPE_F64:
    amm_ndarray_apply_f_unary(double, sin, arr);
    break;
  default:
    fprintf(stderr, "amm_ndarray_sin: unsupported dtype %d\n", arr->dtype);
    return NULL;
  }
  return arr;
}

