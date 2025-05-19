#include "common_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void fill_with_gaussian(float *X, int size) {
  for (int i=0; i<size; i++) X[i] = sqrt(-2.0 * log((float)rand() / RAND_MAX)) * cos(2.0 * M_PI * (float)rand() / RAND_MAX);
}

// [TODO] 多分別のToplevelを作る
int main() {
  // Maddness Workflow
  // 1, Prototype Learning
  // Initialize matrix sampled from gaussian dist.
  int M = 128, N = 128, LDX = 128;
  float *X_Dummy = (float *)malloc(M * N * sizeof(float));
  fill_with_gaussian(X_Dummy, M * N);
  printf("X_Dummy: %f\n", X_Dummy[0]); // the seed is fixed right?
  // 2, Encoding Function
  // 3, Table Construction
  // 4, Aggregation
}
