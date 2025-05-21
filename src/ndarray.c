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

int amm_axis_compute_index_on_memory(Axis* axis, int position) {
  if (axis->random_access_idx == NULL) {
    // Strided Access
    return (axis->offset + position) * axis->stride;
  } else {
    // Random Access
    return ((int*)axis->random_access_idx)[position] * axis->stride;
  }
}
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
// ~~ Shape Solver ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
__amm_give Shape* amm_shape_merge_dims(Shape* shape) {
  // TODO: Gives a simplified shape space.
  
}
// ~~ Apply ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
int any_has_random_access_p(int nrank, int nargs, NDArray** args) {
  for (int i=0; i<nargs; i++) {
    if (args[i]->shape->axes[nrank]->random_access_idx != NULL) return 1;
  }
  return 0;
}
// TODO: Optimize _amm_step_simulated_loop correctly
// 1. Work Correctly and tests are passing in CI
// 2. Optimize the computation of index with regard to index_placeholder caching
// 3. Ensure tests are passing.
void _amm_step_simulated_loop(int current_rank, int nrank, const int* iteration_space, int nargs, NDArray** args,
                              int** index_placeholder, int* offsets, int* offsets_tmp, int* increments
                              ) {
  // Index_Placeholder[NTH_ARG, NTH_RANK] -> args[nth_arg][compute_memory_index(index_placeholder[nth_rank])]
  // Note: we keep index_placeholder because it will be replaced w/ each args' offsets in the future.
  amm_assert(0 <= nargs && nargs <= 3, "Apply only supports 1, 2, or 3 arguments");
  if (current_rank == nrank) {
    for (int i=0; i<nargs; i++) offsets[i] = 0; // Initialize
    for (int r=0; r<current_rank-1; r++)
      for (int i=0; i<nargs; i++)
        offsets[i] += amm_axis_compute_index_on_memory(args[i]->shape->axes[r], index_placeholder[r][i]);
    if (any_has_random_access_p(current_rank, nargs, args)) {
      // If any of them has random access, we need to use element_applier
      for (int nth=0; nth<iteration_space[current_rank]; nth++) {
        for (int i=0; i<nargs; i++) offsets_tmp[i] = offsets[i] + amm_axis_compute_index_on_memory(args[i]->shape->axes[current_rank], nth);
        // Call the element_applier

      }
    } else {
      // If none of then has random access, we can just use range_applier which is faster.
      for (int i=0; i<nargs; i++) increments[i] = amm_ndarray_stride_of(args[i], current_rank);
      // Call the range_applier
      
    }
  } else {
    // Updating Offsets
    for (int nth_element=0; nth_element<iteration_space[current_rank]; nth_element++) {
      for (int nth_arg=0; nth_arg<nargs; nth_arg++) index_placeholder[current_rank][nth_arg] = nth_element;
      // Here, the rank (i+1) loop is independent of the rank (i) loop so that we have no need to realloc the index_placeholder
      // Move forward
      _amm_step_simulated_loop(current_rank + 1, nrank, iteration_space, nargs, args, index_placeholder, offsets, offsets_tmp, increments);
    }
  }
}

void _amm_ndarray_apply(int nargs, NDArray** args) {
  // Impl:
  // 1. Simplify all shape
  // 2. Select the tallest one
  // 3. Reshape other arrays to match them
  // 4. apply recursively (similar to Caten Shape Solver)
  
  // x = simplified_shape
  // Input: 仮想的なLoopSpaceと，各TensorのSimplifyされたShape
  // for 0..3
  //   for 0..5
  //      ...
  amm_assert(nargs > 0, "Invalid number of arguments");
  int nrank = args[0]->shape->nrank; // temporary! should be optimized further!!
  int* iteration_space = malloc(sizeof(int) * nrank);
  int** index_placeholder = (int**)malloc(sizeof(int*) * nrank);
  amm_assert(iteration_space, "Failed to alloc iteration_space");
  for (int i=0; i<nrank; i++) {
    index_placeholder[i] = malloc(sizeof(int) * nargs);
    amm_assert(index_placeholder[i], "Failed to alloc index_placeholder[%d]", i);
  }
  int* offsets = malloc(sizeof(int) * nargs);
  int* offsets_tmp = malloc(sizeof(int) * nargs);
  int* increments = malloc(sizeof(int) * nargs);
  /*
    iteration_space: 仮想的なLoopSpace
    for 0..3
       for 0..5
         ..
  */
  amm_assert(index_placeholder, "Failed to alloc index_placeholder");
  amm_assert(offsets, "Failed to alloc offsets");
  amm_assert(increments, "Failed to alloc increments");
  for (int i=0; i<nrank; i++) iteration_space[i] = args[0]->shape->axes[i]->size;
  // Start simulating the loop
  _amm_step_simulated_loop(0, nrank, iteration_space, nargs, args, index_placeholder, offsets, offsets_tmp, increments);
  free(iteration_space);
  for (int i=0; i<nrank; i++) free(index_placeholder[i]);
  free(index_placeholder);
  free(offsets);
  free(offsets_tmp);
  free(increments);
}
// ~~ Apply Wrappers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#if defined(AMM_C_GCC_MODE)
__amm_keep NDArray* _amm_ndarray_apply_unary(__amm_take NDArray* out, void (*range_applier)(void*, int, int, int), void (*element_applier)(void*, int))
#elif defined(AMM_C_BLOCK_MODE)
__amm_keep NDArray* _amm_ndarray_apply_unary(__amm_take NDArray* out, void (^range_applier)(void*, int, int, int), void (^element_applier)(void*, int))
#endif
{
  _amm_ndarray_apply(1, (NDArray*[]){out});
  return out;
}

#if defined(AMM_C_GCC_MODE)
__amm_keep NDArray* _amm_ndarray_apply_binary(__amm_take NDArray* out, __amm_keep NDArray* in, void (*range_applier)(void*, void*, int, int, int, int, int), void (*element_applier)(void*, int, int))
#elif defined(AMM_C_BLOCK_MODE)
__amm_keep NDArray* _amm_ndarray_apply_binary(__amm_take NDArray* out, __amm_keep NDArray* in, void (^range_applier)(void*, void*, int, int, int, int, int), void (^element_applier)(void*, int, int))
#endif
{
  _amm_ndarray_apply(2, (NDArray*[]){in, out});
  return out;
}

#if defined(AMM_C_GCC_MODE)
__amm_keep NDArray* _amm_ndarray_apply_ternary(__amm_take NDArray* out, __amm_keep NDArray* x, __amm_keep NDArray* y, void (*range_applier)(void*, void*, void*, int, int, int, int, int, int, int), void (*element_applier)(void*, int, int, int))
#elif defined(AMM_C_BLOCK_MODE)
__amm_keep NDArray* _amm_ndarray_apply_ternary(__amm_take NDArray* out, __amm_keep NDArray* x, __amm_keep NDArray* y, void (^range_applier)(void*, void*, void*, int, int, int, int, int, int, int), void (^element_applier)(void*, int, int, int))
#endif
{
  _amm_ndarray_apply(3, (NDArray*[]){out, x, y});
  return out;
}
// ~~ Implementations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
__amm_keep NDArray* amm_ndarray_sin(__amm_take NDArray* arr) {
  switch (arr->dtype) { // TODO: Make Switch Macro
  case AMM_DTYPE_F32:
    amm_ndarray_apply_f_unary(float, sinf, arr);
    break;
  case AMM_DTYPE_F64:
    amm_ndarray_apply_f_unary(double, sin, arr);
    break;
  default:
    fprintf(stderr, "amm_ndarray_sin: sin only supports for float and double, getting: %d\n", arr->dtype);
    return NULL;
  }
  return arr;
}

