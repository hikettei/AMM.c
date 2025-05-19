/*
  include/original_maddness.h
  OriginalMaddnessGemm supports the following things:
  - Original Maddness Matrix Multiplication Approximation as proposed by this paper:
  - Which supports only FP32, BF16.
  - Non-differentiable.
*/

#include "amm_dtype.h"
#include "ndarray.h"

typedef struct OriginalMaddnessGemm OriginalMaddnessGemm;
struct OriginalMaddnessGemm {
  int N, M, K; // A[N M] @ B[M K]  
  int LDX; // Column Major or Row Major.
  int C; // Number of Codebooks
  int nsplits; // Number of splits per codebook
  int n_cluster; // Number of clusters (Usually 16)
  NDArray* quantized_lut; NDArray* buckets; NDArray* protos; // TODO: Quantizes into int8_t ~ binary/ternary?
  AMM_DType dtype; // Data type of the input matrix
};

OriginalMaddnessGemm *amm_original_maddness_gemm_alloc(int N, int M, int K, int LDX, int C, int n_cluster, int nsplits, AMM_DType dtype);
void amm_original_maddness_gemm_free(OriginalMaddnessGemm *mgemm);

void amm_om_setAoffline(OriginalMaddnessGemm *mgemm, NDArray* A_offline);
void amm_om_setA(OriginalMaddnessGemm *mgemm, NDArray* A);
void amm_om_setB(OriginalMaddnessGemm *mgemm, NDArray* B);
