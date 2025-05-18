#include "include/kernels/common_encoder.hpp"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <inttypes.h>

/*
TODO:
Float32, BFloat16, Float, Int2 ~ 8に対して
Prototype Learning, LUT Scanを実装
Extern Cとマクロで(SIMDと同じネーミングで)maddness_encode_i8的なのを作る
Naive = Unoptimized Implementation
*/

// Memo: マクロでf32 ~ i8
void encode_m_f32(const float *X, int m, int n, int ldx,
                  int ncodebooks,
                  const uint32_t *splitdims, const int8_t *splitvals, const float *scales, const float *offsets,
                  uint8_t* out) {
  // TODO
  // rewrite w/ new style of arguments
  // 1. Make Stride Configurable (Please refer to gemm API)
  // 2. Make Block Size Configurable
  const int block_nrows = 32;
  const int nsplits_per_codebook = 4;
  const int vals_per_split = 1 << nsplits_per_codebook; // = 16

  assert(nrows % block_nrows == 0); // nrows must be a multiple of 32
  
  int total_nsplits = ncodebooks * nsplits_per_codebook;
  int maxdim = splitdims[0], mindim = splitdims[0];

  for (int i = 1; i < total_nsplits; i++) {
    if (splitdims[i] > (uint32_t)maxdim) maxdim = splitdims[i];
    if (splitdims[i] < (uint32_t)mindim) mindim = splitdims[i];
  }
  assert(mindim >= 0 && maxdim < ncols);
  int64_t nblocks = nrows / block_nrows;
  
  for (int c = 0; c < ncodebooks; c++) {
    int split_base = c * nsplits_per_codebook;
    for (int b = 0; b < nblocks; b++) {
      for (int i = 0; i < block_nrows; i++) {
        int64_t row = (int64_t)b * block_nrows + i;
        int code = 0;
        for (int s = 0; s < nsplits_per_codebook; s++) {
          uint32_t dim = splitdims[split_base + s];
          float x = X[(int64_t)dim * nrows + row];
          float v = x * scales[split_base + s] + offsets[split_base + s];
          // Quantization with saturation (round → clamp to [-128,127])
          int iv = (int)roundf(v);
          if      (iv > 127) iv = 127;
          else if (iv < -128) iv = -128;
          int8_t x_i8 = (int8_t)iv;
          const int8_t* tbl = all_splitvals + vals_per_split * (split_base + s);
          int8_t threshold = tbl[code];
          int bit = (x_i8 > threshold) ? 1 : 0;
          code = (code << 1) | bit;
        }
        out[c * nrows + row] = (uint8_t)code;
      }
    }
  }
}
