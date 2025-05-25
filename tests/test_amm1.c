
#include "original_maddness.h"
#include "amm_dtype.h"
#include "ndarray.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
# include <time.h>
// TODO: Implement Multiple Maddness Algorithm
// 1. OriginalMaddnessGemm
// 2. Differentiable MaddnessGemm
// 3. QuantizedMaddnessGemm (where the input itself is int)

NDArray* randn(int i, int j) {
  return amm_ndarray_randn(amm_make_shape(2, (int[]){i, j}), AMM_DTYPE_F32);
}

void matmul(NDArray* A, NDArray* B, NDArray* out) {
  // A: N x M
  // B: M x K
  // C: N x K
  int N = amm_ndarray_size_of(A, 0);
  int M = amm_ndarray_size_of(A, 1);
  int K = amm_ndarray_size_of(B, 1);

  int lda1 = amm_ndarray_stride_of(A, 0);
  int lda2 = amm_ndarray_stride_of(A, 1);

  int ldb1 = amm_ndarray_stride_of(B, 0);
  int ldb2 = amm_ndarray_stride_of(B, 1);

  int ldc1 = amm_ndarray_stride_of(out, 0);
  int ldc2 = amm_ndarray_stride_of(out, 1);

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < K; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < M; ++k) {
        sum += ((float*)A->storage)[i * lda1 + k * lda2] * ((float*)B->storage)[ k * ldb1 + j * ldb2];
      }
      ((float*)out->storage)[i * ldc1 + j * ldc2] = sum;
    }
  }     
}

// TODO:
// - [ ] Stablize the implementation, no segv.
// - [ ] Optimize setAoffline Training speed by improving impls, utilizing simd.
// - [ ] Optimize LUTs by Ridge (implement ridge.c by only using ndarray.c)
// - [ ] Complete Encoder/Decoder/Gemm
//  - [ ] Revisit the API Design
// - [ ] Optimize FP32 Implementation for AVX2/Metal GPU
int main() {
  // Maddness Workflow
  // 1, Prototype Learning
  // Initialize matrix sampled from gaussian dist.
  OriginalMaddnessGemm* mgemm = amm_original_maddness_gemm_alloc(128, 128, 128, 1, 16, 4, AMM_DTYPE_F32);
  // We are going to approximate A[N M] @ B[M K]
  NDArray *A_offline = randn(mgemm->N, mgemm->M);
  // for debug
  // amm_ndarray_index_components(A_offline);
  NDArray *A         = randn(mgemm->N, mgemm->M);
  NDArray *A_enc     = amm_ndarray_zeros(amm_make_shape(2, (int[]){mgemm->N, mgemm->n_cluster}), AMM_DTYPE_U8);
  
  NDArray *B         = randn(mgemm->M, mgemm->K);
  print_ndarray(A_offline);

  // 1. SET_A_OFFLINE
  clock_t begin = clock();
  amm_om_setAoffline(mgemm, A_offline);
  printf("Offline Training took %f seconds\n", (double)(clock() - begin) / CLOCKS_PER_SEC);
 
  begin = clock();
  amm_om_setA(mgemm, A, A_enc); // Encode A
  printf("Encoding A took %f seconds\n", (double)(clock() - begin) / CLOCKS_PER_SEC);
  
  begin = clock();
  amm_om_setB(mgemm, B); // Encode B
  printf("Encoding B took %f seconds\n", (double)(clock() - begin) / CLOCKS_PER_SEC);

  // Training & LUT Creation Finished
  // Compute Online Maddness GEMM
  NDArray* out = amm_ndarray_zeros(amm_make_shape(2, (int[]){mgemm->N, mgemm->K}), AMM_DTYPE_U8);

  begin = clock();
  amm_om_setA(mgemm, A, A_enc);
  amm_om_gemm(mgemm, A_enc, out);
  printf("Gemm %f seconds\n", (double)(clock() - begin) / CLOCKS_PER_SEC);
  print_ndarray(out);

  NDArray* out2 = amm_ndarray_zeros(amm_make_shape(2, (int[]){mgemm->N, mgemm->K}), AMM_DTYPE_F32);
  begin = clock();
  matmul(A, B, out2);
  printf("Matmul %f seconds\n", (double)(clock() - begin) / CLOCKS_PER_SEC);
  print_ndarray(out2);
        
  amm_original_maddness_gemm_free(mgemm);

  amm_ndarray_free(A_offline); amm_ndarray_free(A); amm_ndarray_free(B);
  
  // 2, Encoding Function
  // 3, Table Construction
  // 4, Aggregation
}
