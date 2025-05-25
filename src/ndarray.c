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
#include <string.h>

#define AMM_C_DEFAULT_MEMORY_ORDER 0
/*
  ndarray.c:
    - dependency free ndarray library which aimed to smallness (NOT SPEED!). If it is fairy fast, it is ok.
    - as for bottleneck operation (e.g.: gemm) users can replace them with OpenBLAS optimized ones (optional)
    - should only be used for implementing the encoder part which is cached to disk, not for the actual computation part!!.
  TODO:
  - [ ] Merge ShapeTracker
  - [ ] Simulated Loop Collapsing optimization
  - [ ] Vectorized/Parallelized apply
  - [ ] Broadcast Auto
  - [ ] Improve the memory management
 */
int amm_axis_compute_index_on_memory(Axis* axis, int position) {
  if (axis->random_access_idx == NULL) {
    // Strided Access
    return (axis->offset + position * axis->by) * axis->stride;
  } else {
    // Random Access
#if defined(AMM_C_SAFE_MODE)
    amm_assert(axis->by == 1, "amm_axis_compute_index_on_memory: random_access_idx is not supported with by != 1");
#endif    
    return ((int*)axis->random_access_idx)[position] * axis->stride;
  }
}

int amm_shape_compute_index_on_memory(Shape* shape, ...) {
  va_list args;
  va_start(args, shape);
  int index = 0;
  for (int i = 0; i < shape->nrank; ++i) index += amm_axis_compute_index_on_memory(shape->axes[i], va_arg(args, int));
  va_end(args);
  return index;
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
    ax->by                = 1;
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

__amm_give Shape* amm_make_shape(int nrank, int* shape) {
#if AMM_C_DEFAULT_MEMORY_ORDER == 0
  return amm_make_row_major_shape(nrank, shape);
#elif AMM_C_DEFAULT_MEMORY_ORDER == 1
  return amm_make_column_major_shape(nrank, shape);
#else
#error "amm_make_shape: invalid memory order"
#endif
}

void amm_axis_free(Axis* axis) {
  if (!axis) return;
  // if (axis->random_access_idx != NULL) free(axis->random_access_idx);
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
// ~~~ Initialization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
__amm_give NDArray* amm_ndarray_zeros(Shape* shape, AMM_DType dtype) {
  if (!shape) {
    fprintf(stderr, "amm_ndarray_zeros: shape is NULL\n");
    return NULL;
  }
  int total_size = 1;
  for (int i = 0; i < shape->nrank; ++i) {
    total_size *= shape->axes[i]->size;
  }
  void* storage = malloc(total_size * amm_dtype_size(dtype));
  if (!storage) {
    fprintf(stderr, "amm_ndarray_zeros: failed to alloc storage\n");
    return NULL;
  }
  memset(storage, 0, total_size * amm_dtype_size(dtype));
  return amm_ndarray_alloc(shape, storage, dtype);
}

#define sample_randn(dtype) (dtype)sqrt(-2.0 * log((dtype)rand() / RAND_MAX)) * cos(2.0 * M_PI * (dtype)rand() / RAND_MAX)
__amm_give NDArray* amm_ndarray_randn(Shape* shape, AMM_DType dtype) {
  NDArray* arr = amm_ndarray_zeros(shape, dtype);
  amm_assert(arr, "amm_ndarray_randn: failed to alloc NDArray");
  switch (dtype) {
  case AMM_DTYPE_F32:
    amm_ndarray_apply_unary(float, x[x_i] = sample_randn(float), arr);
    break;
  case AMM_DTYPE_F64:
    amm_ndarray_apply_unary(double, x[x_i] = sample_randn(double), arr);
    break;
  default:
    fprintf(stderr, "amm_ndarray_randn: unsupported dtype %d\n", dtype);
    amm_ndarray_free(arr);
    return NULL;
  }
  return arr;
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
  //  amm_assert(amm_ndarray_is_contiguous(arr), "Reshape only works for contiguous arrays");
  amm_shape_free(arr->shape); // Replaces the shape
  arr->shape = new_shape;
  return arr;
}

__amm_keep NDArray* amm_ndarray_permute(__amm_take NDArray* arr, ...) {
  va_list args;
  va_start(args, arr);
  int* perm = malloc(sizeof(int) * arr->shape->nrank);
  if (!perm) {
    fprintf(stderr, "amm_ndarray_permute: failed to alloc perm\n");
    return NULL;
  }
  for (int i = 0; i < arr->shape->nrank; ++i) {
    perm[i] = va_arg(args, int);
    amm_assert(perm[i] >= 0 && perm[i] < arr->shape->nrank, "amm_ndarray_permute: invalid perm[%d]=%d", i, perm[i]);
  }
  va_end(args);
  // TODO: Check if the permutation is valid
  if (!arr) return NULL;
  int nrank = amm_ndarray_rank(arr);
  Axis** old_axes = arr->shape->axes;
  Axis** new_axes = malloc(nrank * sizeof *new_axes);
  if (!new_axes) {
    fprintf(stderr, "amm_ndarray_permute: failed to alloc new_axes\n");
    return NULL;
  }
  for (int i = 0; i < nrank; ++i) new_axes[i] = old_axes[perm[i]];
  // replace axes array and free old
  arr->shape->axes = new_axes;
  free(old_axes);
  free(perm);
  return arr;
}

__amm_keep NDArray* amm_ndarray_view_index(__amm_take NDArray* arr, int rank, int new_size, const int* indices) {
  // Reads the indices as a random accessing.
  // indices are assumed to be in the range of [0, arr->shape->axes[rank]->size)
  amm_assert(rank <= arr->shape->nrank && rank >= 0, "amm_ndarray_view_index: invalid rank %d", rank);
  Axis* axis = arr->shape->axes[rank];
  // amm_assert(axis->random_access_idx == NULL, "amm_ndarray_view_index: random_access(random_access(x)) view merge is not implemented yet.");
#if defined(AMM_C_SAFE_MODE)
  int old_size = amm_ndarray_size_of(arr, rank);
  for (int i=0; i<new_size; i++) {
    int new_idx = indices[i];
    amm_assert(new_idx >= 0 && new_idx < old_size, "amm_ndarray_view_index: Cannot access %dth element from %d vector.", new_idx, old_size);
  }
#endif
  axis->random_access_idx = (void*)indices;
  axis->size = new_size;
  // TODO(hikettei): how to design the indices memory management? is there no double free?
  return arr;
}

__amm_keep NDArray* amm_ndarray_expand(__amm_take NDArray* arr, const int* expand) {
  // TODO: Broadcast shapes among arr->shape and expand.
  // Note: we assume arr->nrank and the length of expand are the same.
  amm_assert(arr->shape->is_contiguous, "amm_ndarray_expand: only works for contiguous arrays");
  for (int i=0; i<amm_ndarray_rank(arr); i++) {
    int size = amm_ndarray_size_of(arr, i);
    int expand_to = expand[i];
    if (expand_to == 1) continue; // No need to expand
    amm_assert(size == 1, "amm_ndarray_expand: the size expanded to %d must be one.", expand_to);
    arr->shape->is_contiguous = false; // We are going to change the stride
    // Broadcast array into expand.
    arr->shape->axes[i]->size = expand_to;
    arr->shape->axes[i]->stride = 0;
  }
  return arr;
}

__amm_keep NDArray* amm_ndarray_slice(__amm_take NDArray* arr, int rank, int from, int to, int by) {
  amm_assert(rank >= 0 && rank < amm_ndarray_rank(arr), "amm_ndarray_slice: invalid rank %d", rank);
  // normalize args
  if (from < 0) from = amm_ndarray_size_of(arr, rank) + from;
  if (to < 0) to = amm_ndarray_size_of(arr, rank) + to;
  if (by < 0) { int tmp=to; to=from; from=tmp; by=-by; }
  if (from > to) by = -by;
  
  if (by > 0) amm_assert(from <= to, "amm_ndarray_slice: invalid range %d:%d", from, to);
  else amm_assert(from > to, "amm_ndarray_slice: invalid range %d:%d", from, to);
  
  Axis* axis = arr->shape->axes[rank];
  // amm_assert(axis->random_access_idx == NULL, "amm_ndarray_slice: random_access(random_access(x)) view merge is not implemented yet.");
  axis->size = abs((1+abs(to - from)) / by);
  axis->by = by;
  axis->offset = from;
  axis->random_access_idx = NULL; // Reset random access index
  
  amm_assert(axis->size > 0, "amm_ndarray_slice: invalid shape %d", axis->size);
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
                              int** index_placeholder, int* offsets, int* offsets_tmp, int* increments,
#if defined(AMM_C_GCC_MODE)
                              void (*range_invoker)(int, int*, int*), void (*elwise_invoker)(int*)
#elif defined(AMM_C_BLOCK_MODE)
                              void (^range_invoker)(int, int*, int*), void (^elwise_invoker)(int*)
#endif
                              )
{
  // Index_Placeholder[NTH_ARG, NTH_RANK] -> args[nth_arg][compute_memory_index(index_placeholder[nth_rank])]
  // Note: we keep index_placeholder because it will be replaced w/ each args' offsets in the future.
  amm_assert(0 <= nargs && nargs <= 3, "Apply only supports 1, 2, or 3 arguments");
  if (current_rank == nrank - 1) {
    for (int i=0; i<nargs; i++) offsets[i] = 0; // Initialize
    for (int r=0; r<current_rank; r++)
      for (int i=0; i<nargs; i++)
        offsets[i] += amm_axis_compute_index_on_memory(args[i]->shape->axes[r], index_placeholder[r][i]);
    if (any_has_random_access_p(current_rank, nargs, args)) {
      // If any of them has random access, we need to use element_applier
      for (int nth=0; nth<iteration_space[current_rank]; nth++) {
        for (int i=0; i<nargs; i++) offsets_tmp[i] = offsets[i] + amm_axis_compute_index_on_memory(args[i]->shape->axes[current_rank], nth);
        elwise_invoker(offsets_tmp);
      }
    } else {
      // If none of then has random access, we can just use range_applier which is faster.
      for (int i=0; i<nargs; i++) increments[i] = amm_ndarray_stride_of(args[i], current_rank) * args[i]->shape->axes[current_rank]->by;
      for (int i=0; i<nargs; i++) offsets[i] += amm_axis_compute_index_on_memory(args[i]->shape->axes[current_rank], 0);
      // Call the range_applier
      range_invoker(iteration_space[current_rank], offsets, increments);
    }
  } else {
    // Updating Offsets
    for (int nth_element=0; nth_element<iteration_space[current_rank]; nth_element++) {
      for (int nth_arg=0; nth_arg<nargs; nth_arg++) index_placeholder[current_rank][nth_arg] = nth_element;
      // Here, the rank (i+1) loop is independent of the rank (i) loop so that we have no need to realloc the index_placeholder
      // Move forward
      _amm_step_simulated_loop(current_rank + 1, nrank, iteration_space, nargs, args, index_placeholder, offsets, offsets_tmp, increments, range_invoker, elwise_invoker);
    }
  }
}

void _amm_ndarray_apply(int nargs, NDArray** args,
#if defined(AMM_C_GCC_MODE)
                        void (*range_invoker)(int, int*, int*), void (*elwise_invoker)(int*)
#elif defined(AMM_C_BLOCK_MODE)
  void (^range_invoker)(int, int*, int*), void (^elwise_invoker)(int*)
#endif
  )
{
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
  // TODO: Below uses the first arg shape space.
  for (int i=0; i<nrank; i++) iteration_space[i] = amm_ndarray_size_of(args[0], i);
  // Start simulating the loop
  _amm_step_simulated_loop(0, nrank, iteration_space, nargs, args, index_placeholder, offsets, offsets_tmp, increments,
                           range_invoker, elwise_invoker);
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
  _amm_ndarray_apply(1, (NDArray*[]){out},
                     amm_lambda(void, (int size, int* offsets, int* increments) { range_applier(out->storage, size, offsets[0], increments[0]); }),
                     amm_lambda(void, (int* indices) { element_applier(out->storage, indices[0]); }));
  return out;
}

#if defined(AMM_C_GCC_MODE)
__amm_keep NDArray* _amm_ndarray_apply_binary(__amm_take NDArray* out, __amm_keep NDArray* in, void (*range_applier)(void*, void*, int, int, int, int, int), void (*element_applier)(void*, void*, int, int))
#elif defined(AMM_C_BLOCK_MODE)
  __amm_keep NDArray* _amm_ndarray_apply_binary(__amm_take NDArray* out, __amm_keep NDArray* in, void (^range_applier)(void*, void*, int, int, int, int, int), void (^element_applier)(void*, void*, int, int))
#endif
{
  _amm_ndarray_apply(2, (NDArray*[]){out, in},
                     amm_lambda(void, (int size, int* offsets, int* increments) { range_applier(out->storage, in->storage, size, offsets[0], increments[0], offsets[1], increments[1]); }),
                     amm_lambda(void, (int* indices) { element_applier(out->storage, in->storage, indices[0], indices[1]); }));
  return out;
}

#if defined(AMM_C_GCC_MODE)
__amm_keep NDArray* _amm_ndarray_apply_ternary(__amm_take NDArray* out, __amm_keep NDArray* x, __amm_keep NDArray* y, void (*range_applier)(void*, void*, void*, int, int, int, int, int, int, int), void (*element_applier)(void*, void*, void*, int, int, int))
#elif defined(AMM_C_BLOCK_MODE)
  __amm_keep NDArray* _amm_ndarray_apply_ternary(__amm_take NDArray* out, __amm_keep NDArray* x, __amm_keep NDArray* y, void (^range_applier)(void*, void*, void*, int, int, int, int, int, int, int), void (^element_applier)(void*, void*, void*, int, int, int))
#endif
{
  _amm_ndarray_apply(3, (NDArray*[]){out, x, y},
                     amm_lambda(void, (int size, int* offsets, int* increments) { range_applier(out->storage, x->storage, y->storage, size, offsets[0], increments[0], offsets[1], increments[1], offsets[2], increments[2]); }),
                     amm_lambda(void, (int* indices) { element_applier(out->storage, x->storage, y->storage, indices[0], indices[1], indices[2]); }));
  return out;
}
// ~~ Implementations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#define _DEFINE_MATH_OP(name, op, dop)                                  \
  __amm_keep NDArray* amm_ndarray_##name(__amm_take NDArray* arr) {     \
    switch (arr->dtype)  {                                              \
    case AMM_DTYPE_F32:                                                 \
      amm_ndarray_apply_unary(float, x[x_i] = op(x[x_i]), arr);         \
      break;                                                            \
    case AMM_DTYPE_F64:                                                 \
      amm_ndarray_apply_unary(double, x[x_i] = dop(x[x_i]), arr);       \
      break;                                                            \
    default:                                                            \
      fprintf(stderr, "amm_ndarray_" #name ": " #name " does not support %d\n", arr->dtype); \
      return NULL;                                                      \
    }                                                                   \
    return arr;                                                         \
  }                                                                     \

_DEFINE_MATH_OP(sin, sinf, sin)
     _DEFINE_MATH_OP(cos, cosf, cos)
     _DEFINE_MATH_OP(tan, tanf, tan)
     _DEFINE_MATH_OP(asin, asinf, asin)
     _DEFINE_MATH_OP(acos, acosf, acos)
     _DEFINE_MATH_OP(atan, atanf, atan)
     _DEFINE_MATH_OP(sinh, sinhf, sinh)
     _DEFINE_MATH_OP(cosh, coshf, cosh)
     _DEFINE_MATH_OP(tanh, tanhf, tanh)
     _DEFINE_MATH_OP(asinh, asinhf, asinh)
     _DEFINE_MATH_OP(acosh, acoshf, acosh)
     _DEFINE_MATH_OP(atanh, atanhf, atanh)
     _DEFINE_MATH_OP(exp, expf, exp)
_DEFINE_MATH_OP(log, logf, log)
_DEFINE_MATH_OP(log10, log10f, log10)
_DEFINE_MATH_OP(log2, log2f, log2)
_DEFINE_MATH_OP(log1p, log1pf, log1p)
_DEFINE_MATH_OP(sqrt, sqrtf, sqrt)
_DEFINE_MATH_OP(cbrt, cbrtf, cbrt)
_DEFINE_MATH_OP(abs, fabsf, fabs)

#define _DEFINE_ARITHMETIC_OP(name, op)                                 \
  __amm_keep NDArray* amm_ndarray_##name(__amm_take NDArray* out, __amm_keep NDArray* x) { \
    switch (out->dtype) {                                               \
    case AMM_DTYPE_F32:                                                 \
      amm_ndarray_apply_binary(float, float, out[out_i] op x[x_i], out, x); \
      break;                                                            \
    case AMM_DTYPE_F64:                                                 \
      amm_ndarray_apply_binary(double, double, out[out_i] op x[x_i], out, x); \
      break;                                                            \
    case AMM_DTYPE_I8:                                                  \
      amm_ndarray_apply_binary(int8_t, int8_t, out[out_i] op x[x_i], out, x); \
      break;                                                            \
    case AMM_DTYPE_I16:                                                 \
      amm_ndarray_apply_binary(int16_t, int16_t, out[out_i] op x[x_i], out, x); \
      break;                                                            \
    case AMM_DTYPE_I32:                                                 \
      amm_ndarray_apply_binary(int32_t, int32_t, out[out_i] op x[x_i], out, x); \
      break;                                                            \
    case AMM_DTYPE_I64:                                                 \
      amm_ndarray_apply_binary(int64_t, int64_t, out[out_i] op x[x_i], out, x); \
      break;                                                            \
    case AMM_DTYPE_U8:                                                  \
      amm_ndarray_apply_binary(uint8_t, uint8_t, out[out_i] op x[x_i], out, x); \
      break;                                                            \
    case AMM_DTYPE_U16:                                                 \
      amm_ndarray_apply_binary(uint16_t, uint16_t, out[out_i] op x[x_i], out, x); \
      break;                                                            \
    case AMM_DTYPE_U32:                                                 \
      amm_ndarray_apply_binary(uint32_t, uint32_t, out[out_i] op x[x_i], out, x); \
      break;                                                            \
    case AMM_DTYPE_U64:                                                 \
      amm_ndarray_apply_binary(uint64_t, uint64_t, out[out_i] op x[x_i], out, x); \
      break;                                                            \
    default:                                                            \
      fprintf(stderr, "amm_ndarray_" #name ": " #name " does not support %d\n", out->dtype); \
      return NULL;                                                      \
    }                                                                   \
    return out;                                                         \
  }                                                                     \


_DEFINE_ARITHMETIC_OP(add, +=)
_DEFINE_ARITHMETIC_OP(sub, -=)
_DEFINE_ARITHMETIC_OP(mul, *=)
_DEFINE_ARITHMETIC_OP(div, /=)
_DEFINE_ARITHMETIC_OP(move, =)
// TODO: maximum, minimum, etc.

__amm_give NDArray* amm_ndarray_ascontiguous(__amm_keep NDArray* arr) {
  int order = AMM_C_DEFAULT_MEMORY_ORDER; // TODO: Make order configurable
  // order = 0 -> row-major, 1 -> column-major
  amm_assert(order == 0 || order == 1, "amm_ndarray_ascontiguous: invalid order %d (must be 0=row-major, 1=column-major)", order);
  Shape* shape;
  int* size_tmp = malloc(arr->shape->nrank * sizeof(int));
  for (int i = 0; i < arr->shape->nrank; ++i) {
    size_tmp[i] = arr->shape->axes[i]->size;
  }
  if (order == 0) {
    shape = amm_make_row_major_shape(arr->shape->nrank, size_tmp);
  } else {
    shape = amm_make_column_major_shape(arr->shape->nrank, size_tmp);
  }
  free(size_tmp);
  NDArray* new_arr = amm_ndarray_zeros(shape, arr->dtype);
  amm_ndarray_move(new_arr, arr);
  return new_arr;
}

__amm_keep NDArray* amm_ndarray_index_components(__amm_take NDArray* arr) {
  switch (arr->dtype) {
  case AMM_DTYPE_I32: 
    amm_ndarray_apply_unary(int, x[x_i] = x_i, arr);
    break;
  case AMM_DTYPE_F32:
    amm_ndarray_apply_unary(float, x[x_i] = x_i, arr);
    break;
  default:
    fprintf(stderr, "amm_ndarray_index_components for dtype %d is not implemented yet.", arr->dtype);
    return NULL;
  }
    
  return arr;
}

// TODO: #define DEFINE_REDUCE(name, binary_op, initial_value)
__amm_give NDArray* amm_ndarray_sum(__amm_keep NDArray* arr, int rank) {
  NDArray* carr = amm_ndarray_ascontiguous(arr);
  int* reduced_size = malloc(carr->shape->nrank * sizeof(int));
  int* expand_to = malloc(carr->shape->nrank * sizeof(int));
  for (int r = 0; r < amm_ndarray_rank(carr); ++r) {
    if (r == rank) reduced_size[r] = 1;
    else reduced_size[r] = amm_ndarray_size_of(carr, r);
    if (r == rank) expand_to[r] = amm_ndarray_size_of(carr, r);
    else expand_to[r] = 1;
  }
  NDArray* out = amm_ndarray_zeros(amm_make_shape(carr->shape->nrank, reduced_size), arr->dtype);
  amm_ndarray_expand(out, expand_to);
  amm_ndarray_add(out, carr);
  amm_ndarray_reshape(out, amm_make_shape(carr->shape->nrank, reduced_size));
  free(reduced_size);
  free(expand_to);
  return out;
}

__amm_give NDArray* amm_ndarray_matmul_naive(__amm_take NDArray* a, __amm_take NDArray* b) {
  // TODO
}

__amm_give NDArray* amm_ndarray_matmul(__amm_take NDArray* a, __amm_take NDArray* b) {
  // TODO: #if defined(AMM_C_USE_OPENBLAS)
}
// ~~ Printers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// recursive print helper
static void print_rec(const NDArray* arr,
                       int nrank,
                       const int* dims,
                      AMM_DType elem_type,
                      int* idx,
                      int level)
{
  if (level == nrank) {
    // compute flat index
    size_t offset = 0;
    for (int d = 0; d < nrank; ++d) {
      offset += amm_axis_compute_index_on_memory(arr->shape->axes[d], idx[d]);
    }
    // print element based on dtype size
    switch (elem_type) {
    case AMM_DTYPE_F32: printf("%f", *((float*)arr->storage+offset)); break;
    case AMM_DTYPE_F64: printf("%f", *((double*)arr->storage+offset)); break;
    case AMM_DTYPE_I8: printf("%hhd", *((int8_t*)arr->storage+offset)); break;
    case AMM_DTYPE_I16: printf("%hd", *((int16_t*)arr->storage+offset)); break;
    case AMM_DTYPE_I32: printf("%d", *((int32_t*)arr->storage+offset)); break;
    case AMM_DTYPE_I64: printf("%llu", *((int64_t*)arr->storage+offset)); break;
    case AMM_DTYPE_U8: printf("%hhu", *((uint8_t*)arr->storage+offset)); break;
    case AMM_DTYPE_U16: printf("%hu", *((uint16_t*)arr->storage+offset)); break;
    case AMM_DTYPE_U32: printf("%u", *((uint32_t*)arr->storage+offset)); break;
    case AMM_DTYPE_U64: printf("%llu", *((uint64_t*)arr->storage+offset)); break;
    default: printf("?");
    }
    return;
  }
  printf("[");
  int dim = dims[level];
  int head = dim, tail = 0;
  if (dim > 20) {
    head = 10;
    tail = 10;
  }
  // print head elements
  for (int i = 0; i < head; ++i) {
    idx[level] = i;
    if (i > 0) printf(", ");
    print_rec(arr, nrank, dims, elem_type, idx, level + 1);
  }
  // ellipsis
  if (dim > 20) {
    printf(", ~");
    for (int i = dim - tail; i < dim; ++i) {
      printf(", ");
      idx[level] = i;
      print_rec(arr, nrank, dims, elem_type, idx, level + 1);
    }
  }
  printf("]");
}
// print_ndarray: prints array contents like numpy, with "~" ellipsis if more than 20 elements
void print_ndarray(__amm_keep NDArray* arr) {
  if (!arr || !arr->shape || !arr->storage) {
    printf("<invalid array>\n");
    return;
  }
  int nrank = amm_ndarray_rank(arr);
  int* dims = malloc(nrank * sizeof *dims);
  int* strides = malloc(nrank * sizeof *dims);
  
  for (int i = 0; i < nrank; ++i) {
    dims[i] = amm_ndarray_size_of(arr, i);
  }

  int* idx = malloc(nrank * sizeof *idx);
  printf("NDArray{shape=[");
  for (int i = 0; i < nrank; ++i) {
    if (i > 0) printf(", ");
    printf("%d", dims[i]);
  }
  printf("], strides=[");
  for (int i = 0; i < nrank; ++i) {
    if (i > 0) printf(", ");
    strides[i] = amm_ndarray_stride_of(arr, i);
    printf("%d", strides[i]);
  }
  printf("], dtype=%d, storage=%p, \n", arr->dtype, arr->storage);
  printf("  data=");
  print_rec(arr, nrank, dims, arr->dtype, idx, 0);
  printf("\n}\n");
  free(idx);
  free(dims);
  free(strides);
}

void amm_assert_shape_eq(__amm_keep NDArray* a, __amm_keep NDArray* b) {
  if (!a || !b) {
    fprintf(stderr, "amm_assert_shape_eq: one of the arrays is NULL\n");
    return;
  }
  if (a->shape->nrank != b->shape->nrank) {
    fprintf(stderr, "amm_assert_shape_eq: rank mismatch %d vs %d\n", a->shape->nrank, b->shape->nrank);
    return;
  }
  for (int i = 0; i < a->shape->nrank; ++i) {
    if (a->shape->axes[i]->size != b->shape->axes[i]->size) {
      fprintf(stderr, "amm_assert_shape_eq: size mismatch at axis %d: %d vs %d\n", i, a->shape->axes[i]->size, b->shape->axes[i]->size);
      return;
    }
  }
}
