#ifdef AMM_C_ALGO_ORIGINAL_MADDNESS

#include "amm_dtype.h"
#include "original_maddness.h"
#include <stdio.h>
#include <stdlib.h>

OriginalMaddnessGemm *amm_original_maddness_gemm_alloc(int N, int M, int K, int LDX, int C, int nsplits, AMM_DType dtype) {
  struct OriginalMaddnessGemm *mgemm = malloc(sizeof *mgemm);
  if (mgemm == NULL) {
    fprintf(stderr, "Failed to allocate memory for OriginalMaddnessGemm\n");
    return NULL;
  }
  mgemm->N = N; mgemm->M = M; mgemm-> K = K;
  mgemm->LDX = LDX;
  mgemm->C = C; mgemm->nsplits = nsplits;
  mgemm->quantized_lut = NULL; // TODO: Initialize quantized LUT if needed
  mgemm->dtype = dtype;
  return mgemm;
}

void amm_original_maddness_gemm_free(OriginalMaddnessGemm *mgemm) {
  if (mgemm != NULL) {
    free(mgemm->quantized_lut); // Free quantized LUT if allocated
    free(mgemm); // Free the main structure
  }
}

void learn_proto_and_hash_function_f32(amm_float32* A_offline, int C, int nsplits) {
  
}
// 1. Prototype Learning
void amm_om_setAoffline_f32(OriginalMaddnessGemm* gemm, amm_float32* A_offline) {
  
}

void amm_om_setA_f32(OriginalMaddnessGemm* gemm, amm_float32* A) {
  
}

void amm_om_setB_f32(OriginalMaddnessGemm* gemm, amm_float32* B) {

}

/*
  Top-level functions for OriginalMaddness
*/
void amm_om_setAoffline(OriginalMaddnessGemm* gemm, void* A_offline) {
  switch (gemm->dtype) {
  case AMM_DTYPE_F32:
    amm_om_setAoffline_f32(gemm, (amm_float32*)A_offline);
    break;
#ifdef AMM_C_USE_BF16
  case AMM_DTYPE_BF16:
    amm_om_setAoffline_bf16(gemm, (amm_bfloat16*)A_offline);
    break;
#endif
  default:
    fprintf(stderr, "Unsupported data type for A_offline\n");
    break;
  }
}

void amm_om_setA(OriginalMaddnessGemm* gemm, void* A) {
  switch (gemm->dtype) {
  case AMM_DTYPE_F32:
    amm_om_setA_f32(gemm, (amm_float32*)A);
    break;
#ifdef AMM_C_USE_BF16
  case AMM_DTYPE_BF16:
    amm_om_setAoffline_bf16(gemm, (amm_bfloat16*)A);
    break
#endif
  default:
      fprintf(stderr, "Unsupported data type for A\n");
    break;
  }
}

void amm_om_setB(OriginalMaddnessGemm* gemm, void* B) {
  switch (gemm->dtype) {
  case AMM_DTYPE_F32:
    amm_om_setA_f32(gemm, (amm_float32*)B);
    break;
#ifdef AMM_C_USE_BF16
  case AMM_DTYPE_BF16:
    amm_om_setAoffline_bf16(gemm, (amm_bfloat16*)B);
    break
#endif
  default:
      fprintf(stderr, "Unsupported data type for B\n");
    break;
  }
}

#endif // AMM_C_ALGO_ORIGINAL_MADDNESS
