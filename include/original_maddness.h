/*
  include/original_maddness.h
  OriginalMaddnessGemm supports the following things:
  - Original Maddness Matrix Multiplication Approximation as proposed by this paper:
  - Which supports only FP32, BF16.
  - Non-differentiable.
*/

#include "amm_dtype.h"

typedef struct {
  int N, M, K; // A[N M] @ B[M K]  
  int LDX; // Column Major or Row Major.
  int C; // Number of Codebooks
  int nsplits; // Number of splits per codebook
  void* quantized_lut; // TODO: Quantizes into int8_t ~ binary/ternary?
} OriginalMaddnessGemm;

// Naming Convention:
// amm_<algorithm>_<operation>_<dtype> where:
// - algorithm is namely **O**riginal**M**addness
void amm_om_setAoffline_f32(OriginalMaddnessGemm* mgemm, amm_float32* A_Offline);
void amm_om_setA_f32(OriginalMaddnessGemm* mgemm, amm_float32* A);
void amm_om_setB_f32(OriginalMaddnessGemm* mgemm, amm_float32* B);

#ifdef AMM_C_USE_BF16

void amm_om_setAoffline_bf16(OriginalMaddnessGemm* mgemm, amm_bfloat16* A_Offline);
void amm_om_setA_bf16(OriginalMaddnessGemm* mgemm, amm_bfloat16* A);
void amm_om_setB_bf16(OriginalMaddnessGemm* mgemm, amm_bfloat16* B);

#endif
