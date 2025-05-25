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
#include <float.h>
#include <tgmath.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
// ~~ Alloc/Free ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
OriginalMaddnessGemm *amm_original_maddness_gemm_alloc(int N, int M, int K, int LDX, int C, int nsplits, AMM_DType dtype) {
  struct OriginalMaddnessGemm *mgemm = malloc(sizeof *mgemm);
  if (mgemm == NULL) {
    fprintf(stderr, "Failed to allocate memory for OriginalMaddnessGemm\n");
    return NULL;
  }
  mgemm->N = N; mgemm->M = M; mgemm-> K = K;
  mgemm->LDX = LDX;
  mgemm->C = C; mgemm->nsplits = nsplits; mgemm->n_cluster = 2 << (nsplits-1);
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
  bucket->threshold_candidates_count = 0;
  bucket->threshold_candidates = NULL;
  bucket->left_child = NULL; bucket->right_child=NULL;
  bucket->indices = NULL;
  bucket->n_indices = 0;
  return bucket;
}

void amm_bucket_threshold_candidate_push(Bucket* bucket, float c) {
  if (bucket->threshold_candidates_count == 0) {
    bucket->threshold_candidates = malloc(sizeof(float));
  } else {
    bucket->threshold_candidates = realloc(bucket->threshold_candidates, (bucket->threshold_candidates_count + 1) * sizeof(float));
  }
  if (bucket->threshold_candidates == NULL) {
    fprintf(stderr, "Failed to allocate memory for threshold candidates\n");
    return;
  }
  ((float*)bucket->threshold_candidates)[bucket->threshold_candidates_count] = c;
  bucket->threshold_candidates_count++;              
}

void amm_bucket_free(Bucket* bucket) {
  if (bucket != NULL) {
    if (bucket->left_child) amm_bucket_free(bucket->left_child);
    if (bucket->right_child) amm_bucket_free(bucket->right_child);
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
  argsort((float*)sliced->storage, amm_ndarray_size_of(x, 0), (int*)sorted->storage, -1);
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
  
  NDArray* x_sort_indices = sort_rows_based_on_col(a_offline_r1, dim);
  NDArray* x_sort_indices_rev = ndarray_reverse(x_sort_indices);
  
  int N = amm_ndarray_size_of(x_sort_indices, 0);
  int D = amm_ndarray_size_of(A_offline, 1);

  NDArray* x_head = amm_ndarray_zeros(amm_make_shape(2, (int[]){N, D}), A_offline->dtype);
  NDArray* x_tail = amm_ndarray_zeros(amm_make_shape(2, (int[]){N, D}), A_offline->dtype);

  amm_ndarray_view_index(a_offline_r1, 0, N, (int*)x_sort_indices->storage);
  NDArray* a_offline_r2 = amm_ndarray_ascontiguous(a_offline_r1);
  cumulative_sse(a_offline_r2, x_head);
  amm_ndarray_view_index(a_offline_r1, 0, N, (int*)x_sort_indices_rev->storage);
  NDArray* a_offline_r3 = amm_ndarray_ascontiguous(a_offline_r1);
  cumulative_sse(a_offline_r3, x_tail);
  amm_ndarray_add(x_head, x_tail);
  NDArray* s_out = amm_ndarray_sum(x_head, 1);

  int N1 = amm_ndarray_size_of(s_out, 0);
  NDArray* s_out_i = amm_ndarray_zeros(amm_make_shape(1, (int[]){N1}), AMM_DTYPE_I32);
  argsort((float*)s_out->storage, N1, (int*)s_out_i->storage, -1);
  int best_idx = ((int*)s_out_i->storage)[0];
  int next_idx = MIN(1+best_idx, amm_ndarray_size_of(a_offline_r, 0) - 1);

  amm_assert(best_idx >= 0 && best_idx < amm_ndarray_size_of(x_sort_indices, 0), "compute_optimal_val_splits: wrong best_idx? %d", best_idx);
  amm_assert(next_idx >= 0 && next_idx < amm_ndarray_size_of(x_sort_indices, 0), "compute_optimal_val_splits: wrong next_idx? %d", next_idx);
  int col_idx1 = amm_ndarray_aref(int, x_sort_indices, best_idx);
  int col_idx2 = amm_ndarray_aref(int, x_sort_indices, next_idx);
  amm_assert(col_idx1 >= 0 && col_idx1 < amm_ndarray_size_of(a_offline_r, 0), "compute_optimal_val_splits: wrong col_idx1? %d", col_idx1);
  amm_assert(col_idx2 >= 0 && col_idx2 < amm_ndarray_size_of(a_offline_r, 0), "compute_optimal_val_splits: wrong col_idx1? %d", col_idx2);

  float val1 = amm_ndarray_aref(float, a_offline_r, col_idx1, dim);
  float val2 = amm_ndarray_aref(float, a_offline_r, col_idx2, dim);

  threshold[0] = (val1 + val2) / 2;
  loss[0] = amm_ndarray_aref(float, s_out, best_idx);

  amm_ndarray_free(x_head); amm_ndarray_free(x_tail);
  amm_ndarray_free(a_offline_r3);
  amm_ndarray_free(a_offline_r2);
  amm_ndarray_free(a_offline_r);
  amm_ndarray_free(s_out);
  amm_ndarray_free(x_sort_indices);
  amm_ndarray_free(x_sort_indices_rev);
}

int optimal_val_splits(NDArray* A_offline, Bucket* bucket, NDArray* total_losses, int d, int dim, int tree_level) {
  if (bucket->tree_level == tree_level) {
    float* threshold = malloc(sizeof(float));
    float* loss = malloc(sizeof(float));
    compute_optimal_val_splits(threshold, loss, A_offline, bucket, dim); // slow
    float threshold_ = threshold[0], loss_ = loss[0];
    free(threshold); free(loss);
    amm_ndarray_aref(float, total_losses, 0, d) += loss_;
    float curr_loss = amm_ndarray_aref(float, total_losses, 0, d);
    amm_bucket_threshold_candidate_push(bucket, threshold_);
    if (d == 0) { // TODO: Introduce early judge?
      return 0;
    } else {
      for (int i=0; i<amm_ndarray_size_of(total_losses, 1); i++)
        if (amm_ndarray_aref(float, total_losses, 0, i) >= curr_loss) return 0;
      return 1;
    }
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

void learn_quantized_params(Bucket* bucket, NDArray* A_offline, int best_dim) {
  // Appendix B
  NDArray* sorts = sort_rows_based_on_col(A_offline, best_dim);
  int idx1 = ((int*)sorts->storage)[0];
  int idx2 = ((int*)sorts->storage)[amm_ndarray_size_of(sorts, 0) - 1];
  float min_loss = amm_ndarray_aref(float, A_offline, idx1, best_dim);
  float max_loss = amm_ndarray_aref(float, A_offline, idx2, best_dim);
  amm_ndarray_free(sorts);
  float min_val = -FLT_MAX;
  float max_val = FLT_MAX;
  for (int i=0; i<bucket->threshold_candidates_count; i++) {
    float c = ((float*)bucket->threshold_candidates)[i];
    min_val = MIN(min_val, c);
    max_val = MAX(max_val, c);
  }
  float offset = (min_loss + min_val) / 2.0f;
  float upper_val = ((max_loss + max_val) / 2.0f) - offset;
  
  float l = log2f(254.0f / upper_val);
  float scale = powf(2.0f, l);
  
  bucket->scale = scale;
  bucket->offset = offset;
  bucket->threshold_quantized = (int)roundf((bucket->threshold - offset) * scale);
}

void bucket_map_tree(Bucket* bucket, int target_tree_level,
#if defined(AMM_C_GCC_MODE)
                     void (*f)(Bucket*)
#else
                     void (^f)(Bucket*)
#endif
                     ) {
  if (bucket->tree_level == target_tree_level) f(bucket);
  else {
    if (bucket->left_child) bucket_map_tree(bucket->left_child, target_tree_level, f);
    if (bucket->right_child) bucket_map_tree(bucket->right_child, target_tree_level, f);
  }
}

void optimize_split_thresholds(Bucket* bucket, int min_idx, int best_dim, int nth_split, NDArray* A_offline) {
  if (bucket->tree_level == nth_split) {
    bucket->index = best_dim;
    bucket->threshold = ((float*)bucket->threshold_candidates)[min_idx];
    learn_quantized_params(bucket, A_offline, best_dim);
  }

  Bucket* left = bucket->left_child;
  Bucket* right = bucket->right_child;
  if (left && right) {
    optimize_split_thresholds(left, min_idx, best_dim, nth_split, A_offline);
    optimize_split_thresholds(right, min_idx, best_dim, nth_split, A_offline);
  }
}

NDArray* tflist_as_index_list(NDArray* arr) {
  int size = amm_ndarray_size_of(arr, 0);
  int* tflist = (int*)arr->storage;
  int required_size = 0;
  for (int i=0; i<size; i++) required_size += (tflist[i] == 1 ? 1 : 0);

  if (required_size == 0) {
    NDArray* ret = amm_ndarray_zeros(amm_make_shape(1, (int[]){size}), AMM_DTYPE_I32);
    amm_ndarray_index_components(ret);
    return ret;
  }
  
  int* index_list = malloc(required_size * sizeof(int));
  int c = 0;
  for (int i=0; i<size; i++)
    if (tflist[i] == 1.0) {
      index_list[c] = i;
      c++;
    }
  return amm_ndarray_alloc(amm_make_shape(1, (int[]){required_size}), index_list, AMM_DTYPE_I32);
}

Bucket* create_new_bucket(NDArray* points, int lv, int idx) {
  Bucket* new_bucket = amm_bucket_alloc();
  new_bucket->indices = points->storage;
  new_bucket->n_indices = amm_ndarray_size_of(points, 0);
  new_bucket->tree_level = lv;
  new_bucket->id = idx;
  return new_bucket;
}

void optimize_bucket_splits(Bucket* bucket, int best_dim, NDArray* A_offline) {
  int right_idx = bucket->id * 2 + 1;
  int left_idx = bucket->id * 2;
  NDArray* A_offline_cp = amm_ndarray_ascontiguous(A_offline);
  amm_ndarray_view_index(A_offline_cp, 0, bucket->n_indices, bucket->indices);
  amm_ndarray_slice(A_offline_cp, 1, best_dim, best_dim, 1);
  float threshold = bucket->threshold;

  int mask_size[1] = {bucket->n_indices};
  NDArray* left_mask = amm_ndarray_zeros(amm_make_shape(1, mask_size), AMM_DTYPE_I32);
  NDArray* right_mask = amm_ndarray_zeros(amm_make_shape(1, mask_size), AMM_DTYPE_I32);

  amm_ndarray_apply_binary(int, float, out[out_i] = x[x_i] > threshold ? 1 : 0, left_mask, A_offline_cp);
  amm_ndarray_apply_binary(int, float, out[out_i] = x[x_i] > threshold ? 0 : 1, right_mask, A_offline_cp);
  
  NDArray* left_side_points = tflist_as_index_list(left_mask);
  NDArray* right_side_points = tflist_as_index_list(right_mask);
  amm_ndarray_free(A_offline_cp);
  // Mwmo: sum(left_mask) == 0.0 case is really required?
  if (bucket->left_child == NULL && bucket->right_child == NULL) {
    bucket->left_child = create_new_bucket(left_side_points, bucket->tree_level+1, left_idx);
    bucket->right_child = create_new_bucket(right_side_points, bucket->tree_level+1, right_idx);
  } else {
    bucket->left_child->n_indices = amm_ndarray_size_of(left_side_points, 0);
    bucket->left_child->indices = left_side_points->storage;
    
    bucket->right_child->n_indices = amm_ndarray_size_of(right_side_points, 0);
    bucket->right_child->indices = right_side_points->storage;
    
    optimize_bucket_splits(bucket->left_child, best_dim, A_offline);
    optimize_bucket_splits(bucket->right_child, best_dim, A_offline);
  }
  amm_ndarray_free(left_mask); amm_ndarray_free(right_mask);
}

Bucket* learn_binary_tree_splits(NDArray* A_offline, NDArray* col_losses, int col_i, int steps, int nsplits) {
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
    argsort((float*)col_losses->storage, steps, (int*)col_losses_i->storage, 1); // col_losses_i <- argosrt(col_losses)
    amm_ndarray_apply_unary(float, x[x_i] = 0.0f, total_losses); // TODO: Implement amm_ndarray_fill
    // Optimize splits based on col_losses
    for (int d=0; d<steps; d++)
      for (int lv=0; lv<=nth_split; lv++)
        if (optimal_val_splits(A_offline, bucket, total_losses, d, ((int*)col_losses_i->storage)[d], lv) != 0) break;
    float min_tmp = FLT_MIN;
    int min_idx;
    for (int i=0; i<steps; i++) min_tmp = MIN(min_tmp, ((float*)total_losses->storage)[i]);
    for (int i=0; i<steps; i++) if (((float*)total_losses->storage)[i] == min_tmp) min_idx = i;
    int best_dim = ((int*)col_losses_i->storage)[min_idx];

    optimize_split_thresholds(bucket, min_idx, best_dim, nth_split, A_offline);
    optimize_bucket_splits(bucket, best_dim, A_offline);
  }
  
  amm_ndarray_free(col_losses_i); amm_ndarray_free(total_losses);
  return bucket;
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
  int nrows = amm_ndarray_size_of(A_offline, 0);
  // Allocation
  if (gemm->buckets == NULL) gemm->buckets = malloc(sizeof(Bucket*) * gemm->M / steps);
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
    gemm->buckets[nth] = learn_binary_tree_splits(A_offline, col_losses, col_i, steps, gemm->nsplits);
    NDArray* centroids = amm_ndarray_zeros(amm_make_shape(2, (int[]){1, gemm->M}), AMM_DTYPE_F32);
    bucket_map_tree(gemm->buckets[nth], gemm->nsplits,
                    amm_lambda(void, (Bucket* buck) {
                        amm_assert(buck->n_indices > 0, "buck->id %d", buck->id);
                        amm_ndarray_apply_unary(float, x[x_i] = 0.0f, centroids);
                        
                        NDArray* m = amm_ndarray_sum(A_offline, 0);
                        amm_ndarray_apply_unary(float, x[x_i] /= (float)buck->n_indices, m);
                        
                        amm_ndarray_slice(centroids, 1, col_i, col_i+steps-1, 1);
                        
                        amm_ndarray_move(centroids, m);
                        amm_ndarray_slice(centroids, 1, 0, gemm->M, 1);
                        amm_ndarray_view_index(A_offline, 0, buck->n_indices, buck->indices);
                        amm_ndarray_expand(centroids, (int[]){buck->n_indices, 1});
                        amm_ndarray_sub(A_offline, centroids);
                        amm_ndarray_slice(A_offline, 0, 0, nrows-1, 1);
                        amm_assert(buck->id >= 0 && buck->id < gemm->n_cluster, "The bucket id %d is out range of [0, %d).", buck->id, gemm->n_cluster);
                        amm_ndarray_slice(gemm->protos, 0, nth, nth, 1);
                        amm_ndarray_slice(gemm->protos, 1, buck->id, buck->id, 1);
                        amm_ndarray_reshape(centroids, amm_make_shape(3, (int[]){1, 1, gemm->M}));
                        amm_ndarray_move(gemm->protos, centroids);
                        amm_ndarray_reshape(centroids, amm_make_shape(2, (int[]){1, gemm->M}));
                        amm_ndarray_free(m);
                      }));
    amm_ndarray_free(centroids);
  }
  amm_ndarray_slice(gemm->protos, 0, 0, gemm->C-1, 1);
  amm_ndarray_slice(gemm->protos, 1, 0, gemm->n_cluster-1, 1);
  amm_ndarray_slice(gemm->protos, 2, 0, gemm->M-1, 1);
  // reset slice of protos
  amm_ndarray_free(col_losses);
}

void flatten_bucket_params(Bucket** buckets, int n_buckets_in_gemm, int nsplits, float* offsets, float* scales, int* dims, int* qts) {
  int n_buckets_per_row = 0;
  for (int i=0;i<nsplits;i++) n_buckets_per_row = (2 << i) + n_buckets_per_row;
  amm_assert(offsets == NULL && scales == NULL && dims == NULL && qts == NULL,
             "flatten_bucket_params: offsets, scales, dims, qts must be allocated before calling this function");
  offsets = malloc(sizeof(float) * n_buckets_per_row * n_buckets_in_gemm);
  scales = malloc(sizeof(float) * n_buckets_per_row * n_buckets_in_gemm);
  dims = malloc(sizeof(int) * n_buckets_per_row * n_buckets_in_gemm);
  qts  = malloc(sizeof(int) * n_buckets_per_row * n_buckets_in_gemm);
  int offset = 0;
  for (int i=0; i<nsplits; i++) {
    for (int b=0; b<n_buckets_in_gemm; b++) {
      Bucket* bucket = buckets[b];
      bucket_map_tree(bucket, i, amm_lambda(void, (Bucket* buck) {
            offsets[offset + buck->id] = buck->offset;
            scales[offset + buck->id] = buck->scale;
            dims[offset + buck->id] = buck->index;
            qts[offset + buck->id] = buck->threshold_quantized;
          }));
      offset += (2 << i);
    }
  }
}

void learn_proto_and_hash_function(OriginalMaddnessGemm* gemm, NDArray* A_offline) {
  init_and_learn_offline(gemm, A_offline); // gemm.buckets = new_bucket; gemm.protos = new_proto;
  // TODO
  // - [ ] Implement optimize-ridge function.
}
// 1. Prototype Learning
void amm_om_setAoffline(OriginalMaddnessGemm* gemm, NDArray* A_offline) {
  learn_proto_and_hash_function(gemm, A_offline);
  // Convert bucket threshold, dim, quantized offsets/scale into NDArray.
  // TODO: Store into DISK.
  float* offsets = NULL, *scales = NULL;
  int* dims = NULL, *qts = NULL;
  flatten_bucket_params(gemm->buckets, gemm->C, gemm->nsplits, offsets, scales, dims, qts);
  
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void amm_om_setA(OriginalMaddnessGemm* gemm, NDArray* A) {
  // mithral-encode-fp32-t
}

void amm_om_setB(OriginalMaddnessGemm* gemm, NDArray* B) {
  // scan and compute lut
}

#endif // AMM_C_ALGO_ORIGINAL_MADDNESS
