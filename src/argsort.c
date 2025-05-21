#include "argsort.h"
#include <stdio.h>
#include <stdlib.h>
// TODO: Move to ndarray.c
static int cmp_pair(const void *a, const void *b) {
    const Pair *pa = (const Pair*)a;
    const Pair *pb = (const Pair*)b;
    if (pa->val < pb->val) return -1;
    if (pa->val > pb->val) return +1;
    return 0;
}

void argsort(const float *arr, int n, int *idxs) {
    Pair *pairs = malloc(n * sizeof(Pair));
    for (int i = 0; i < n; ++i) {
        pairs[i].idx = i;
        pairs[i].val = arr[i];
    }
    qsort(pairs, n, sizeof(Pair), cmp_pair);
    for (size_t i = 0; i < n; ++i) idxs[i] = pairs[i].idx;
    free(pairs);
}
