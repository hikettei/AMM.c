#ifdef AMM_C_ALGO_ORIGINAL_MADDNESS

#include "amm_dtype.h"
#include "original_maddness.h"
#include "utils.h"
#include "argsort.h"

#ifdef AMM_C_USE_OMP
#include <omp.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
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
// Bucket
Bucket *amm_bucket_alloc() {
  Bucket *bucket = malloc(sizeof(Bucket));
  if (bucket == NULL) {
    fprintf(stderr, "Failed to allocate memory for Bucket\n");
    return NULL;
  }
  bucket->tree_level = 0; bucket->id = 0;
  bucket->scale = 0.0f; bucket->offset = 0.0f;
  bucket->threshold_quantized = 0;
  bucket->index = 0;
  bucket->threshold = 0.0f;
  bucket->threshold_candidates = NULL;
  bucket->children = NULL;
  bucket->indices = NULL;
  bucket->n_indices = 0;
  return bucket;
}

void amm_bucket_free(Bucket* bucket) {
  if (bucket != NULL) {
    free(bucket->threshold_candidates);
    free(bucket->children);
    free(bucket->indices);
    free(bucket);
  }
}

Bucket *amm_bucket_alloc_toplevel(int N) {    
  Bucket* bucket = amm_bucket_alloc();
  bucket->indices = malloc(N * sizeof(int));
  bucket->n_indices = N;
  for (int i=0; i<N; i++) ((int*)bucket->indices)[i] = i;
  return bucket;
}

void amm_bucket_col_variances(float* out_storage, Bucket* bucket, NDArray* A_offline, int col_offset, int steps, bool scale) {
  for (int nth=0; nth<A_offline->shape[0]; nth++) {
    int row_idx = ((int*)bucket->indices)[nth];
    float mu = 0.0f; // Accumlator for column-wise sum
    for (int c=0; c<steps; c++) mu += ((float*)A_offline->storage)[row_idx * A_offline->strides[0] + (col_offset + c) * A_offline->strides[1]];
    mu /= steps; // Mean
    // Broadcast for shape[1]
    float xi;
    for (int c=0; c<steps; c++) {
      xi = ((float*)A_offline->storage)[row_idx * A_offline->strides[0] + (col_offset + c) * A_offline->strides[1]];
      xi -= mu; xi *= xi; // (xi - mu)^2
      if (scale) xi /= steps; // (xi - mu)^2 / steps
      out_storage[c] += xi; // Result
    }
  }
}

void sumup_col_sqs(float* col_losses, Bucket* bucket, NDArray* A_offline, int col_i, int steps);
void sumup_col_sqs(float* col_losses, Bucket* bucket, NDArray* A_offline, int col_i, int steps) {
   if (bucket->indices != NULL) amm_bucket_col_variances(col_losses, bucket, A_offline, col_i, steps, 0);

  if (bucket->children != NULL) {
    sumup_col_sqs(col_losses, ((Bucket**)bucket->children)[0], A_offline, col_i, steps);
    sumup_col_sqs(col_losses, ((Bucket**)bucket->children)[1], A_offline, col_i, steps);
  }
}

float* compute_optimal_val_splits(NDArray* A_offline, int col_i, int steps, Bucket* bucket, int dim, int dth) {
  if (bucket->indices == NULL || bucket->n_indices < 2) return (float[]){0.0f, 0.0f}; // No split possible
  for (int nrow=0; nrow<A_offline->shape[0]; nrow++) {
    
  }
}

int optimal_val_splits(NDArray* A_offline, int col_i, int steps, Bucket* bucket, float* total_losses,
                       int d, int dth, int lv) {
  if (bucket->tree_level == lv) {
    
  } else {
    amm_assert(bucket->children != NULL, "optimal_val_splits: bucket->children is NULL");
  }
}

void learn_binary_tree_splits(NDArray* A_offline, int col_i, int steps, int nsplits) {
  /*
    
Figure:
              B(1, 1)                  | nth=0
         /----------------\            |
     B(2, 1)            B(2,2)         | nth=1
   /---------\        /---------\      |
B(3, 1)  B(3, 2)   B(3, 3)  B(3, 4)    | nth=2
                                       | ...
                                       | nth=nsplits
  */
  Bucket* bucket = amm_bucket_alloc_toplevel(A_offline->shape[0]); // Start with one big buckets covering all rows
  float* col_losses = malloc(steps * sizeof(float));
  float* total_losses = malloc(steps * sizeof(float));
  int*   col_indices = malloc(steps * sizeof(int));
  {
    amm_noarg_callback reset_col_losses = amm_lambda(void, (void) { for (int i=0; i<steps; i++) col_losses[i] = 0.0f; });
    amm_noarg_callback fill_col_indices_argmax = amm_lambda(void, (void) { argsort(col_losses, steps, col_indices); });
    // Training
    for (int epoch=0; epoch<nsplits; epoch++) {
      reset_col_losses();
      sumup_col_sqs(col_losses, bucket, A_offline, col_i, steps);
      fill_col_indices_argmax();
      for (int c=0;c<steps;c++) {
        printf("col_losses[%d]: %f\n", c, col_losses[c]); // Loss by column, the goal here is to minimize them
        printf("col_indices[%d]: %d\n", c, col_indices[c]); // Indices of the columns
      }
      // Update splits
      for (int d=0; d<steps; d++)
        for (int lv=0; lv<=epoch; lv++)
          if (optimal_val_splits(A_offline, col_i, steps, bucket, total_losses, d, col_indices[d], lv) != 0) break;
      // Find the best split
      
    }
  }
}

// Protoype Learning
void init_and_learn_offline(OriginalMaddnessGemm* gemm, NDArray* A_offline) {
  /*
    The function init_and_learn_offline_fp32 clusters the prototypess from A_offline, and then constructs the encoding function g(a).
    A_offline is a matrix of size [nrows, gemm->M] and the value of nrows can be any number regardless of gemm->N. (i.e.: the more nrows the better cluster)

    This function allocates two arrays:
    - gemm->buckets: [?]
    - gemm->protos:  [C n_cluster gemm->M/gemm->C]

     D          C      C    (where D = 2 * C)
  ++++++       +--    -+-
N ++++++ =>  N +--  N -+-  <- N*D Matrix is disjointed into N*C Matrix.
  ++++++       +--,   -+-  ... * (N/D), Binary-Tree-Split is applied into each visible area.
  */
  amm_assert((gemm->M % gemm->C) == 0, "init_and_learn_offline_fp32: M should be divisible by C");
  int steps = gemm->M / gemm->C;
  // Allocation
  // TODO: The shape of buckets???
  if (gemm->buckets == NULL) gemm->buckets = malloc(gemm->C * gemm->nsplits * sizeof(int)); // [C, nsplits]
  if (gemm->protos == NULL)  gemm->protos = malloc(gemm->C * gemm->n_cluster * steps * sizeof(float)); // [C, n_cluster, M/C]
  // amm_lambda_type(int, (int, int)) aref_buckets = amm_lambda(int, (int i, int j) { return i + lda * j;});

  // Reading A_offline [T, 0:4], A_offline[T, 4:8], A_offline[T, 8:12], ..., A_offline[T, col_i:col_i+4]
#ifdef AMM_C_USE_OMP
#pragma omp parallel for
#endif
  for (int col_i=0, nth=0; col_i<gemm->M; col_i+=steps, nth++) {
    printf("col_i: %d, nth: %d\n", col_i, nth);
    learn_binary_tree_splits(A_offline, col_i, steps, gemm->nsplits);
  }
  // NDArray作る。
  // 短冊状にした各Bucketごとに学習
  // どこからHardware Specificにする？
  // ここからやっていい？
  // 1. INDEX(i, j)を作る
  // 2. Memory Order???
  // Changed policy: Implement HashLearning in C
}

void learn_proto_and_hash_function(OriginalMaddnessGemm* gemm, NDArray* A_offline) {
  init_and_learn_offline(gemm, A_offline); // gemm.buckets = new_bucket; gemm.protos = new_proto;
  
}
// 1. Prototype Learning
void amm_om_setAoffline(OriginalMaddnessGemm* gemm, NDArray* A_offline) {
  learn_proto_and_hash_function(gemm, A_offline);
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void amm_om_setA(OriginalMaddnessGemm* gemm, NDArray* A) {
  
}

void amm_om_setB(OriginalMaddnessGemm* gemm, NDArray* B) {

}

#endif // AMM_C_ALGO_ORIGINAL_MADDNESS
