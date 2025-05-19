#pragma once
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
// (TODO) More Follows ...
