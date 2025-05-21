#include "amm_dtype.h"
#include <stdbool.h>
#include "utils.h"
#pragma once

#ifndef AMM_UTILS_H
#error "ndarray.h: either of AMM_C_GCC_MODE or AMM_C_BLOCK_MODE must be defined by loading utils.h"
#endif

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
  int offset; // TODO: Introduce [3:5] (i.e.: real_size and X[-1:-3]) reversing.
  int stride;
  void* random_access_idx; // If random_access_idx was set to int* pointer, ndarray will read the corresponding index by random_access_idx[0], ... random_access_idx[size-1]
};

int amm_axis_compute_index_on_memory(Axis* axis, int position);

typedef struct Shape Shape;
struct Shape {
  int nrank;
  Axis** axes;
  // int offset; TODO: Introduce extra offset.
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
int amm_ndarray_rank(__amm_keep const NDArray* arr);
int amm_ndarray_size_of(__amm_keep const NDArray* arr, int dim);
int amm_ndarray_stride_of(__amm_keep const NDArray* arr, int dim);
int amm_ndarray_total_size(__amm_keep const NDArray* arr);
bool amm_ndarray_is_contiguous(__amm_keep const NDArray* arr);
// ~~~ Initializers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
__amm_give NDArray* amm_ndarray_zeros(Shape* shape, AMM_DType dtype);
__amm_give NDArray* amm_ndarray_randn(Shape* shape, AMM_DType dtype);
// TODO: __amm_give NDArray* amm_ndarray_copy(__amm_keep NDArray* arr);
// ~~~ Movements ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// TODO: The only operation here is apply_map, (this can implement even matmul, im2col, which is enough for our goal)
__amm_keep NDArray* amm_ndarray_reshape(__amm_take NDArray* arr, Shape* new_shape);
__amm_keep NDArray* amm_ndarray_permute(__amm_take NDArray* arr, const int* perm);
__amm_keep NDArray* amm_ndarray_view(__amm_take NDArray* arr, int* shape);
__amm_keep NDArray* amm_ndarray_expand(__amm_take NDArray* arr, int* expand);
// ~~~ Apply ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// TODO: Add Optimization using OpenMP depending on the hardware
#define amm_expand_applier_unary(dtype, op)                             \
  amm_lambda(void, (void* x_, int size, int x_offset, int incx) {       \
      for (int n=0; n<size; n++) {                                      \
        int x_i = x_offset + n * incx;                                  \
        dtype* x = (dtype*)x_;                                          \
        op; }})                                                         \

#define amm_expand_applier_binary(dtype_out, dtype_in, op)              \
  amm_lambda(void, (void* out_, void* x_, int size, int out_offset, int inco, int x_offset, int incx) { \
      for (int n=0; n<size; n++) {                                      \
        int out_i = out_offfset + n * inco;                             \
        int x_i = x_offset + n * incx;                                  \
        dtype_out* out = (dtype_out*)out_;                              \
        dtype_in* x = (dtype_in*)x_;                                    \
        op; }})                                                         \

#define amm_expand_applier_ternary(dtype_out, dtype_in1, dtype_in2, op) \
  amm_lambda(void, (dtype_out* out_, dtype_in1* x_, dtype_in2* y), int size, int out_offset, int inco, int x_offset, int incx, int y_offset, int incy) { \
    for (int n=0; n<size; n++) {                                        \
      int out_i = out_offfset + n * inco;                               \
      int x_i = x_offset + n * incx;                                    \
      int y_i = y_offset + n * incy;                                    \
      dtype_out* out = (dtype_out*)out_;                                \
      dtype_in1* x = (dtype_in1*)x_;                                    \
      dtype_in2* y = (dtype_in2*)y_;                                    \
      op; }})                                                           \

#define amm_ndarray_apply_f_unary(dtype, f, arr) _amm_ndarray_apply_unary(arr, amm_expand_applier_unary(dtype, x[x_i] = f(x[x_i])), amm_lambda(void, (void* x, int index) { ((dtype*)x)[index] = f(((dtype*)x)[index]); }))
#define amm_ndarray_apply_f_binary(dtype_out, dtype_in, f, out_arr, in_arr) _amm_ndarray_apply_binary(out_arr, in_arr, amm_expand_applier_binary(dtype_out, dtype_in, x[out_i] = f(x[x_i])), amm_lambda(void, (void* out, void* x, int out_i, int x_i) { ((dtype_out*)out)[out_i] = f(((dtype_in*)x)[x_i]); }))
#define amm_ndarray_apply_f_ternary(dtype_out, dtype_in1, dtype_in2, f, out_arr, in_arr1, in_arr2) _amm_ndarray_apply_ternary(out_arr, in_arr1, in_arr2, amm_expand_applier_ternary(dtype_out, dtype_in1, dtype_in2, x[out_i] = f(x[x_i], y[y_i]), amm_lambda(void, (void* out, void* x, void* y, int out_i, int x_i, int y_i) { ((dtype_out*)out)[out_i] = f(((dtype_in1*)x)[x_i], ((dtype_in2*)y)[y_i]); })))

#define amm_ndarray_apply_unary(dtype, form, arr) _amm_ndarray_apply_unary(arr, amm_expand_applier_unary(dtype, form), amm_lambda(void, (void* x_, int x_i) { dtype* x = (dtype*)x_; form; }));
#define amm_ndarray_apply_binary(dtype_out, dtype_in, form, out_arr, in_arr) _amm_ndarray_apply_binary(out_arr, in_arr, amm_expand_applier_binary(dtype_out, dtype_in, form), amm_lambda(void, (void* out_, void* x_, int out_i, int x_i) { dtype_out* out = (dtype_out*)out_; dtype_in* x = (dtype_in*)x_; form; }));
#define amm_ndarray_apply_ternary(dtype_out, dtype_in1, dtype_in2, form, out_arr, in_arr1, in_arr2) _amm_ndarray_apply_ternary(out_arr, in_arr1, in_arr2, amm_expand_applier_ternary(dtype_out, dtype_in1, dtype_in2, form), amm_lambda(void, (void* out_, void* x_, void* y_, int out_i, int x_i, int y_i) { dtype_out* out = (dtype_out*)out_; dtype_in1* x = (dtype_in1*)x_; dtype_in2* y = (dtype_in2*)y_; form; }));

// __amm_keep NDArray* _amm_ndarray_apply();

#if defined(AMM_C_GCC_MODE)
__amm_keep NDArray* _amm_ndarray_apply_unary(__amm_take NDArray* out, void (*range_applier)(void*, int, int, int), void (*element_applier)(void*, int));
__amm_keep NDArray* _amm_ndarray_apply_binary(__amm_take NDArray* out, __amm_keep NDArray* in, void (*range_applier)(void*, void*, int, int, int, int, int), void (*element_applier)(void*, void*, int, int));
__amm_keep NDArray* _amm_ndarray_apply_ternary(__amm_take NDArray* out, __amm_keep NDArray* x, __amm_keep NDArray* y, void (*range_applier)(void*, void*, void*, int, int, int, int, int, int, int), void (*element_applier)(void*, void*, void*, int, int, int));
#elif defined(AMM_C_BLOCK_MODE)
__amm_keep NDArray* _amm_ndarray_apply_unary(__amm_take NDArray* out, void (^range_applier)(void*, int, int, int), void (^element_applier)(void*, int));
__amm_keep NDArray* _amm_ndarray_apply_binary(__amm_take NDArray* out, __amm_keep NDArray* in, void (^range_applier)(void*, void*, int, int, int, int, int), void (^element_applier)(void*, void*, int, int));
__amm_keep NDArray* _amm_ndarray_apply_ternary(__amm_take NDArray* out, __amm_keep NDArray* x, __amm_keep NDArray* y, void (^range_applier)(void*, void*, void*, int, int, int, int, int, int, int), void (^element_applier)(void*, void*, void*, int, int, int));
#endif
// ~~ Operations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
__amm_keep NDArray* amm_ndarray_sin(__amm_take NDArray* arr);

__amm_keep NDArray* amm_ndarray_index_components(__amm_take NDArray* arr);
// TODO:
// - ndarray_cast
// - ndarray_arange

// TODO: Use either of OpenBLAS or Our implementation
// __amm_keep NDArray* _amm_ndarray_matmul_naive(); <- wrapping and make it testable.
// __amm_keep NDArray* amm_ndarray_matmul();

// #define amm_ndarray_apply_binary(applier, arr)
// Implement Cast how to?
// TODO: Create optimizer version which if the array is contiguous then converted to 1d op;
// TODO: 再起的にIndexを計算
// TODO: Contiguous Partを見つけたら，BLAS_LIKE Operationでvectorizeとかして計算

void print_ndarray(__amm_keep NDArray* arr);
