#pragma once
#include <stdint.h>
#include <stdio.h>
typedef enum {
  AMM_DTYPE_F32 = 0,
  AMM_DTYPE_F64 = 1,
  AMM_DTYPE_BF16 = 2,
  AMM_DTYPE_I8 = 3,
  AMM_DTYPE_I16 = 4,
  AMM_DTYPE_I32 = 5,
  AMM_DTYPE_I64 = 6,
  AMM_DTYPE_U8 = 7,
  AMM_DTYPE_U16 = 8,
  AMM_DTYPE_U32 = 9,
  AMM_DTYPE_U64 = 10,
} AMM_DType;

#define amm_float64 double
#define amm_float32 float
#ifdef AMM_C_USE_BF16
  #define amm_bfloat16 __fp16 // todo: dousuru?
#endif

int amm_dtype_size(AMM_DType dtype) {
  switch (dtype) {
  case AMM_DTYPE_F32: return sizeof(amm_float32);
  case AMM_DTYPE_F64: return sizeof(amm_float64);
#ifdef AMM_C_USE_BF16
  case AMM_DTYPE_BF16: return sizeof(amm_bfloat16);
#endif
  case AMM_DTYPE_I8: return sizeof(int8_t);
  case AMM_DTYPE_I16: return sizeof(int16_t);
  case AMM_DTYPE_I32: return sizeof(int32_t);
  case AMM_DTYPE_I64: return sizeof(int64_t);
  case AMM_DTYPE_U8: return sizeof(uint8_t);
  case AMM_DTYPE_U16: return sizeof(uint16_t);
  case AMM_DTYPE_U32: return sizeof(uint32_t);
  case AMM_DTYPE_U64: return sizeof(uint64_t);
  default: {
    fprintf(stderr, "Unknown data type: %d\n", dtype);
    return -1;
  }
  }
}
