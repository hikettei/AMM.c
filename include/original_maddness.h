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

typedef struct Bucket Bucket;
struct Bucket {
  int tree_level; int id;
  float scale; float offset; // Quantization Parameters (TODO: Abstraction for Quantization? QuantizedNDArray?)
  int threshold_quantized;
  int index;
  float threshold;
  int threshold_candidates_count;
  void* threshold_candidates;
  Bucket* left_child; Bucket* right_child;
  void* indices;
  int n_indices; // Number of indices in this bucket
};

Bucket *amm_bucket_alloc();
void amm_bucket_free(Bucket* bucket);

void amm_om_setAoffline(OriginalMaddnessGemm *mgemm, NDArray* A_offline);
void amm_om_setA(OriginalMaddnessGemm *mgemm, NDArray* A);
void amm_om_setB(OriginalMaddnessGemm *mgemm, NDArray* B);
