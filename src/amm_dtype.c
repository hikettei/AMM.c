#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "amm_dtype.h"

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
