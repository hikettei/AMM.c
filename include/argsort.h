typedef struct {
    int idx;
    double val;
} Pair;

static int cmp_pair(const void *a, const void *b);
void argsort(const float* arr, int n, int* idxs);
