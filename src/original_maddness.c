#ifdef AMM_C_ALGO_ORIGINAL_MADDNESS

#include "amm_dtype.h"
#include "original_maddness.h"

#include <stdio.h>
#include <stdlib.h>
#include <utils.h>
// ~~ Alloc/Free ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
OriginalMaddnessGemm *amm_original_maddness_gemm_alloc(int N, int M, int K, int LDX, int C, int n_cluster, int nsplits, AMM_DType dtype) {
  struct OriginalMaddnessGemm *mgemm = malloc(sizeof *mgemm);
  if (mgemm == NULL) {
    fprintf(stderr, "Failed to allocate memory for OriginalMaddnessGemm\n");
    return NULL;
  }
  mgemm->N = N; mgemm->M = M; mgemm-> K = K;
  mgemm->LDX = LDX;
  mgemm->C = C; mgemm->nsplits = nsplits; mgemm->n_cluster = n_cluster;
  // TODO: func* allocator = {... if dtype = ...}
  mgemm->quantized_lut = NULL; mgemm->buckets = NULL; mgemm->protos = NULL; // TODO: Initialize quantized LUT if needed
  mgemm->dtype = dtype;
  return mgemm;
}

void amm_original_maddness_gemm_free(OriginalMaddnessGemm *mgemm) {
  if (mgemm != NULL) {
    free(mgemm->quantized_lut); free(mgemm->buckets); free(mgemm->protos);
    free(mgemm); // Free the main structure
  }
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void init_and_learn_offline_fp32(OriginalMaddnessGemm* gemm, amm_float32* A_offline, int nrows, int lda) {
  /*
    The function init_and_learn_offline_fp32 clusters the prototypess from A_offline, and then constructs the encoding function g(a).
    A_offline is a matrix of size [nrows, gemm->M] and the value of nrows can be any number regardless of gemm->N. (i.e.: the more nrows the better cluster)

    This function allocates two arrays:
    - gemm->buckets: [?]
    - gemm->protos:  [C n_cluster gemm->M/gemm->C]

   D         C      C
  +++       +--    -+-
N +++ =>  N +--  N -+-  <- N*D Matrix is disjointed into N*C Matrix.
  +++       +--,   -+-  ... * (N/D), Binary-Tree-Split is applied into each visible area.
  */
  amm_assert((gemm->M % gemm->C) == 0, "init_and_learn_offline_fp32: M should be divisible by C");
  int steps = gemm->M / gemm->C;
  // Allocation
  // TODO: The shape of buckets???
  if (gemm->buckets == NULL) gemm->buckets = malloc(gemm->C * gemm->nsplits * sizeof(int)); // [C, nsplits]
  if (gemm->protos == NULL)  gemm->protos = malloc(gemm->C * gemm->n_cluster * steps * sizeof(float)); // [C, n_cluster, M/C]

  // Reading A_offline [T, 0:4], A_offline[T, 4:8], A_offline[T, 8:12], ...
  // 短冊状にした各Bucketごとに学習
  // どこからHardware Specificにする？
  // ここからやっていい？
  // 1. INDEX(i, j)を作る
  // 2. Memory Order???
}

void learn_proto_and_hash_function_f32(OriginalMaddnessGemm* gemm, amm_float32* A_offline, int nrows, int lda) {
  init_and_learn_offline_fp32(gemm, A_offline, nrows, lda); // gemm.buckets = new_bucket; gemm.protos = new_proto;
  
}
// 1. Prototype Learning
void amm_om_setAoffline_f32(OriginalMaddnessGemm* gemm, amm_float32* A_offline, int nrows, int lda) {
  learn_proto_and_hash_function_f32(gemm, A_offline, nrows, lda);
}

void amm_om_setA_f32(OriginalMaddnessGemm* gemm, amm_float32* A) {
  
}

void amm_om_setB_f32(OriginalMaddnessGemm* gemm, amm_float32* B) {

}

/*
  Top-level functions for OriginalMaddness
*/
void amm_om_setAoffline(OriginalMaddnessGemm* gemm, void* A_offline, int nrows, int lda) {
  switch (gemm->dtype) {
  case AMM_DTYPE_F32:
    amm_om_setAoffline_f32(gemm, (amm_float32*)A_offline, nrows, lda);
    break;
#ifdef AMM_C_USE_BF16
  case AMM_DTYPE_BF16:
    amm_om_setAoffline_bf16(gemm, (amm_bfloat16*)A_offline, nrows, lda);
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
