#include "common_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float* randn(int size) {
  float *x = (float *)malloc(size * sizeof(float));
  for (int i=0; i<size; i++) x[i] = sqrt(-2.0 * log((float)rand() / RAND_MAX)) * cos(2.0 * M_PI * (float)rand() / RAND_MAX);
  return x;
}

// [TODO] 多分別のToplevelを作る
int main() {
  // Maddness Workflow
  // 1, Prototype Learning
  // Initialize matrix sampled from gaussian dist.
  int N = 512, M = 512, K = 512;
  int LDX = 512; // Column Major
  int C = 16; // Number of Codebooks
  int nsplits = 4; // Number of splits per codebook

  // We are going to approximate A[N M] @ B[M K]
  float *A_Offline = randn(M * N);
  float *A         = randn(M * N);
  // 1. SET_A_OFFLINE
  
  
  // 2. SET_A
  // 3. SET_B

  
  // 2, Encoding Function
  // 3, Table Construction
  // 4, Aggregation
}
