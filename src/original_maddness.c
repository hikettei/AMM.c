#ifdef AMM_C_ALGO_ORIGINAL_MADDNESS

#include "amm_dtype.h"
#include "ndarray.h"
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
  bucket->left_child = NULL; bucket->right_child=NULL;
  bucket->indices = NULL;
  bucket->n_indices = 0;
  return bucket;
}

void amm_bucket_free(Bucket* bucket) {
  if (bucket != NULL) {
    free(bucket->threshold_candidates);
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

void col_variances(NDArray* out, Bucket* bucket, NDArray* A_offline, int do_mean_p) {
  NDArray* a_offline_r = amm_ndarray_ascontiguous(A_offline);
  // Slicing a_offling_r rows by a jurisdictions of the bucket.
  amm_ndarray_view_index(a_offline_r, 0, bucket->n_indices, bucket->indices);
  int D = amm_ndarray_size_of(a_offline_r, 1);
  NDArray* mu = amm_ndarray_sum(a_offline_r, 1); // TODO: Replace w/ mean
  amm_ndarray_apply_unary(float, x[x_i] /= (float)D, mu);
  amm_ndarray_expand(mu, (int[]){1, D});
  amm_ndarray_sub(a_offline_r, mu); // (a_offline_r - mu)^2
  amm_ndarray_mul(a_offline_r, a_offline_r);
  if (do_mean_p == 1) amm_ndarray_apply_unary(float, x[x_i] /= (float)bucket->n_indices, a_offline_r);
  NDArray* caf = amm_ndarray_ascontiguous(a_offline_r);
  amm_ndarray_free(a_offline_r);
  NDArray* result = amm_ndarray_sum(caf, 0);
  amm_assert_shape_eq(out, result);
  amm_ndarray_apply_binary(float, float, out[out_i] += x[x_i], out, result);
  amm_ndarray_free(result);
  amm_ndarray_free(caf);
  amm_ndarray_free(mu);
}

void sumup_col_sum_sqs(NDArray* col_losses, Bucket* bucket, NDArray* A_offline) {
  if (bucket->indices) col_variances(col_losses, bucket, A_offline, 0);
  if (bucket->left_child != NULL && bucket->right_child != NULL) {
    sumup_col_sum_sqs(col_losses, bucket->left_child, A_offline);
    sumup_col_sum_sqs(col_losses, bucket->right_child, A_offline);
  }
}

void learn_binary_tree_splits(NDArray* A_offline, NDArray* col_losses, int col_i, int steps, int nsplits) {
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
  // A_offline: [n_rows, steps] sliced array.
  // B(1, 1) is the origin of all buckets
  amm_assert(amm_ndarray_size_of(A_offline, 1) == steps, "invaild size of A_offline");
  Bucket* bucket = amm_bucket_alloc_toplevel(amm_ndarray_size_of(A_offline, 0)); // Start with one big buckets covering all rows
  for (int nth_split=0; nth_split < nsplits; nth_split++) {
    amm_ndarray_apply_unary(float, x[x_i] = 0.0f, col_losses); // TODO: Implement amm_ndarray_fill
    sumup_col_sum_sqs(col_losses, bucket, A_offline);
    // argsort col losses
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
  amm_assert(amm_ndarray_rank(A_offline) == 2, "init_and_learn_offline_fp32: A_offline must be 2d ndarray");
  amm_assert((gemm->M % gemm->C) == 0, "init_and_learn_offline_fp32: M should be divisible by C");
  int steps = gemm->M / gemm->C;
  // Allocation
  // TODO: The shape of buckets???
  if (gemm->buckets == NULL) gemm->buckets = amm_ndarray_zeros(amm_make_shape(2, (int[]){gemm->C, gemm->nsplits}), A_offline->dtype);
  if (gemm->protos == NULL)  gemm->protos = amm_ndarray_zeros(amm_make_shape(3, (int[]){gemm->C, gemm->n_cluster, gemm->M}), A_offline->dtype);

  NDArray* col_losses = amm_ndarray_zeros(amm_make_shape(2, (int[]){1, steps}), AMM_DTYPE_F32);
  
#ifdef AMM_C_USE_OMP
#pragma omp parallel for
#endif
  for (int col_i=0, nth=0; col_i<gemm->M; col_i+=steps, nth++) {
    printf("col_i: %d, nth: %d\n", col_i, nth);
    // Reading A_offline [T, 0:4], A_offline[T, 4:8], A_offline[T, 8:12], ..., A_offline[T, col_i:col_i+4] ...
    // as well as prototype
    amm_ndarray_slice(A_offline, 1, col_i, col_i+steps, 1);
    amm_ndarray_slice(gemm->protos, 2, col_i, col_i+steps, 1);
    learn_binary_tree_splits(A_offline, col_losses, col_i, steps, gemm->nsplits);
  }

  amm_ndarray_free(col_losses);
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
