
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
  int a_rows = 1024;
  // We are going to approximate A[N M] @ B[M K]
  NDArray *A_offline = randn(a_rows, mgemm->M);
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
  
  amm_original_maddness_gemm_free(mgemm);

  amm_ndarray_free(A_offline); amm_ndarray_free(A); amm_ndarray_free(B);
  
  // 2, Encoding Function
  // 3, Table Construction
  // 4, Aggregation
}
