#include "amm_dtype.h"
#include "original_maddness.h"
#include <stdio.h>
#include <stdlib.h>

OriginalMaddnessGemm *amm_original_maddness_gemm_alloc(int N, int M, int K, int LDX, int C, int nsplits) {
  struct OriginalMaddnessGemm *mgemm = malloc(sizeof *mgemm);
  if (mgemm == NULL) {
    fprintf(stderr, "Failed to allocate memory for OriginalMaddnessGemm\n");
    return NULL;
  }
  mgemm->N = N; mgemm->M = M; mgemm-> K = K;
  mgemm->LDX = LDX;
  mgemm->C = C; mgemm->nsplits = nsplits;
  mgemm->quantized_lut = NULL; // TODO: Initialize quantized LUT if needed
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
