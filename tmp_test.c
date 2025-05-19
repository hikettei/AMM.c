
#include "original_maddness.h"
#include "amm_dtype.h"
#include "ndarray.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// TODO: Implement Multiple Maddness Algorithm
// 1. OriginalMaddnessGemm
// 2. Differentiable MaddnessGemm
// 3. QuantizedMaddnessGemm (where the input itself is int)

// メモ: より新しい実装が発見された時に，今までのソースツリーから非依存で追加できるようにしたい。
// Toplevelは完全に別にしよう。

// テスト用 ~~~~~~~~
// 後で移動する
NDArray* randn(int i, int j) {
  int size = i * j;
  float *x = (float *)malloc(size * sizeof(float));
  for (int i=0; i<size; i++) x[i] = sqrt(-2.0 * log((float)rand() / RAND_MAX)) * cos(2.0 * M_PI * (float)rand() / RAND_MAX);
  return amm_ndarray_alloc(2, (int[]){i, j}, (int[]){j, 1}, &x, AMM_DTYPE_F32);
}

// TODO:
int main() {
  // Maddness Workflow
  // 1, Prototype Learning
  // Initialize matrix sampled from gaussian dist.
  OriginalMaddnessGemm* mgemm = amm_original_maddness_gemm_alloc(512, 512, 512, 1, 16, 16, 4, AMM_DTYPE_F32);
  int a_rows = 1024;
  // We are going to approximate A[N M] @ B[M K]
  NDArray *A_offline = randn(a_rows, mgemm->N);
  NDArray *A         = randn(mgemm->N, mgemm->M);
  NDArray *B         = randn(mgemm->M, mgemm->K);
  // 1. SET_A_OFFLINE
  amm_om_setAoffline(mgemm, A_offline);

  // Compute A_hat @ B
  amm_om_setA(mgemm, A);
  amm_om_setB(mgemm, B);
  
  amm_original_maddness_gemm_free(mgemm);

  amm_ndarray_free(A_offline); amm_ndarray_free(A); amm_ndarray_free(B);
        
  // 2. SET_A
  // 3. SET_B
  
  // 2, Encoding Function
  // 3, Table Construction
  // 4, Aggregation
}
