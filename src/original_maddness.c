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
#include <string.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
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

__amm_give NDArray* sort_rows_based_on_col(__amm_keep NDArray* x, int dim) {
  NDArray* x1 = amm_ndarray_ascontiguous(x);
  amm_ndarray_slice(x1, 1, dim, dim, 1);
  NDArray* sliced = amm_ndarray_ascontiguous(x1);
  NDArray* sorted = amm_ndarray_zeros(amm_make_shape(1, (int[]){amm_ndarray_size_of(x, 0)}), AMM_DTYPE_I32);
  argsort((float*)sliced->storage, amm_ndarray_size_of(x, 0), (int*)sorted->storage);
  int N = amm_ndarray_size_of(x1, 0);
  amm_ndarray_apply_unary(int, x[x_i] = N - x[x_i], sorted)
  amm_ndarray_free(sliced);
  amm_ndarray_free(x1);
  return sorted;
}

__amm_give NDArray* ndarray_reverse(__amm_keep NDArray* x) {
  NDArray* x1 = amm_ndarray_ascontiguous(x);
  amm_ndarray_slice(x1, 0, 0, amm_ndarray_size_of(x1, 0)-1, -1);
  NDArray* x2 = amm_ndarray_ascontiguous(x1);
  amm_ndarray_free(x1);
  return x2;
}

float *cumulative_sse(NDArray* xp, NDArray* cumsses) {
  int N = amm_ndarray_size_of(xp, 0);
  int D = amm_ndarray_size_of(xp, 1);
  
  float *cumX  = calloc(D, sizeof(float));
  float *cumX2 = calloc(D, sizeof(float));
  if (!cumX || !cumX2) {
    free(cumX); free(cumX2);
    return NULL;
  }

  float *x = malloc(sizeof(float) * N * D);
  if (!x) {
    free(cumX); free(cumX2);
    return NULL;
  }
  memcpy(x, (float*)xp->storage, sizeof(float) * N * D);
  memcpy(cumX, x, sizeof(float) * D);
  for (int j = 0; j < D; ++j) cumX2[j] = cumX[j] * cumX[j];
  for (int i = 0; i < N; ++i) {
    float *x_row = x + (size_t)i * D;
    float  lr    = 1.0 / (2.0 + i);
    for (int j = 0; j < D; ++j) {
      cumX[j]  += x_row[j];
      cumX2[j] += x_row[j];
    }
    for (int j = 0; j < D; ++j) {
      float meanX = cumX[j] * lr;
      float mx    = meanX * cumX[j];
      mx           = -mx;
      ((float*)cumsses->storage)[(size_t)i * D + j] = cumX2[j] + mx;
    }
  }
  free(cumX); free(cumX2); free(x);
}

void compute_optimal_val_splits(float* threshold, float* loss, NDArray* A_offline, Bucket* bucket, int dim) {
  if (bucket->indices == NULL || bucket->n_indices < 2) {
    threshold[0] = 0.0, loss[0] = 0.0;
    return;
  }

  NDArray* a_offline_r = amm_ndarray_ascontiguous(A_offline);
  amm_ndarray_view_index(a_offline_r, 0, bucket->n_indices, bucket->indices);
  NDArray* a_offline_r1 = amm_ndarray_ascontiguous(a_offline_r);
  amm_ndarray_free(a_offline_r);
  NDArray* x_sort_indices = sort_rows_based_on_col(a_offline_r1, dim);
  NDArray* x_sort_indices_rev = ndarray_reverse(x_sort_indices);
  int N = amm_ndarray_size_of(x_sort_indices, 0);
  int D = amm_ndarray_size_of(A_offline, 1);
  // tmp
  NDArray* x_head = amm_ndarray_zeros(amm_make_shape(2, (int[]){N, D}), A_offline->dtype);
  NDArray* x_tail = amm_ndarray_zeros(amm_make_shape(2, (int[]){N, D}), A_offline->dtype);
  
  amm_ndarray_view_index(a_offline_r1, 0, N, (int*)x_sort_indices->storage);
  NDArray* a_offline_r2 = amm_ndarray_ascontiguous(a_offline_r1);
  cumulative_sse(a_offline_r2, x_head);
  amm_ndarray_free(a_offline_r2);
  amm_ndarray_view_index(a_offline_r1, 0, N, (int*)x_sort_indices_rev->storage);
  NDArray* a_offline_r3 = amm_ndarray_ascontiguous(a_offline_r1);
  cumulative_sse(a_offline_r3, x_tail);
  amm_ndarray_free(a_offline_r3);
  amm_ndarray_add(x_head, x_tail);
  NDArray* s_out = amm_ndarray_sum(x_head, 1);
  amm_ndarray_free(x_head); amm_ndarray_free(x_tail);

  int N1 = amm_ndarray_size_of(s_out, 0);
  NDArray* s_out_i = amm_ndarray_zeros(amm_make_shape(1, (int[]){N1}), AMM_DTYPE_I32);
  argsort((float*)s_out->storage, N1, (int*)s_out_i->storage);
  amm_ndarray_apply_unary(int, x[x_i] = N1 - x[x_i], s_out_i);
  int best_idx = ((int*)s_out_i->storage)[0];
  int next_idx = MIN(1+best_idx, amm_ndarray_size_of(A_offline, 0) - 1);

  int col_idx1 = ((int*)x_sort_indices->storage)[best_idx];
  int col_idx2 = ((int*)x_sort_indices->storage)[next_idx];

  float val1 = amm_ndarray_aref(float, A_offline, col_idx1, dim);
  float val2 = amm_ndarray_aref(float, A_offline, col_idx2, dim);

  threshold[0] = (val1 + val2) / 2;
  loss[0] = amm_ndarray_aref(float, s_out, best_idx);
  printf("threshold:%f loss:%f\n", threshold[0], loss[0]);

  //amm_ndarray_free(x_sort_indices);
  //amm_ndarray_free(x_sort_indices_rev);
  //  amm_ndarray_free(s_out);
}

int optimal_val_splits(NDArray* A_offline, Bucket* bucket, NDArray* total_losses, int d, int dim, int tree_level) {
  if (bucket->tree_level == tree_level) {
    float* threshold = malloc(sizeof(float));
    float* loss = malloc(sizeof(float));
    compute_optimal_val_splits(threshold, loss, A_offline, bucket, dim);
    float threshold_ = threshold[0], loss_ = loss[0];
    free(threshold); free(loss);
  } else {
    Bucket* left = bucket->left_child;
    Bucket* right = bucket->right_child;
    amm_assert(left != NULL && right != NULL, "optimal_val_splits: could not find any buckets?...");
    int res1 = optimal_val_splits(A_offline, left, total_losses, d, dim, tree_level);
    int res2 = optimal_val_splits(A_offline, right, total_losses, d, dim, tree_level);
    int res3 = optimal_val_splits(A_offline, bucket, total_losses, d, dim, bucket->tree_level); // Compute current-level node.
    return res1 || res2 || res3;
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
  NDArray* col_losses_i = amm_ndarray_zeros(amm_make_shape(1, (int[]){steps}), AMM_DTYPE_I32);
  NDArray* total_losses = amm_ndarray_zeros(amm_make_shape(2, (int[]){1, steps}), AMM_DTYPE_F32);
  for (int nth_split=0; nth_split < nsplits; nth_split++) {
    amm_ndarray_apply_unary(float, x[x_i] = 0.0f, col_losses); // TODO: Implement amm_ndarray_fill
    sumup_col_sum_sqs(col_losses, bucket, A_offline);
    argsort((float*)col_losses->storage, steps, (int*)col_losses_i->storage); // col_losses_i <- argosrt(col_losses)
    amm_ndarray_apply_unary(float, x[x_i] = 0.0f, total_losses); // TODO: Implement amm_ndarray_fill
    // Optimize splits based on col_losses
    for (int d=0; d<steps; d++)
      for (int lv=0; lv<=nth_split; lv++)
        if (optimal_val_splits(A_offline, bucket, total_losses, d, ((int*)col_losses_i->storage)[d], lv) != 0) break;
    // argsort col losses
  }
  amm_ndarray_free(col_losses_i) ;amm_ndarray_free(total_losses);
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
    amm_ndarray_slice(A_offline, 1, col_i, col_i+steps-1, 1);
    amm_ndarray_slice(gemm->protos, 2, col_i, col_i+steps-1, 1);
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
