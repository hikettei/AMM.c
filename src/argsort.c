#include "argsort.h"
#include <stdio.h>
#include <stdlib.h>
// TODO: Move to ndarray.c
static int cmp_pair(const void *a, const void *b) {
  const Pair *pa = (const Pair*)a;
  const Pair *pb = (const Pair*)b;
  if (pa->val < pb->val) return -1;
  if (pa->val > pb->val) return +1;
  // tie-break: smaller original index first
  if (pa->idx < pb->idx) return -1;
  if (pa->idx > pb->idx) return +1;
  return 0;
}

void argsort(const float *arr, int n, int *idxs, int sign) {
  // -1 to [largest_idx, ... smallest_idx]
  // 1 to [smallest_idx, ... largest_idx]
  if (arr == NULL || idxs == NULL || n <= 0 || (sign != 1 && sign != -1)) return;
  Pair *pairs = malloc((size_t)n * sizeof(Pair));
  if (pairs == NULL) return;

  for (int i = 0; i < n; ++i) {
    pairs[i].idx = i;
    pairs[i].val = arr[i] * (float)sign;
  }
  qsort(pairs, (size_t)n, sizeof(Pair), cmp_pair);
  for (int i = 0; i < n; ++i)  idxs[i] = pairs[i].idx;
  free(pairs);
}
