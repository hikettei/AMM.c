#include "comm_original_maddness.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <inttypes.h>

//  ~~ Encoder ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void encode_m_f32(const float *X, int m, int n, int ldx1, int ldx2,
                  int C, int nsplits,
                  const uint32_t * splitdims, const int8_t *splitvals, const float *scales, const float *offsets,
                  uint8_t* out) {
  int block_nrows = 32;
  int64_t nblocks = m / block_nrows;
  int vals_per_split = 1 << nsplits;
  int total_nsplits = C * nsplits;
  int maxdim = splitdims[0], mindim = splitdims[0];

  assert(m % block_nrows == 0); // m must be a multiple of 32
  
  for (int i = 1; i < total_nsplits; i++) {
    if (splitdims[i] > (uint32_t)maxdim) maxdim = splitdims[i];
    if (splitdims[i] < (uint32_t)mindim) mindim = splitdims[i];
  }
  assert(mindim >= 0 && maxdim < n);
  
  for (int c=0; c<C; c++) {
    int split_base = c * nsplits;
    // Tile over rows
    for (int b=0; b < nblocks; b++) {
      for (int i=0; i<block_nrows; i++) {
        int64_t row = (int64_t)b * block_nrows + i;
        int code = 0;
        for (int s=0; s< nsplits; s++) {
          uint32_t dim = splitdims[split_base + s];
          float x = X[(int64_t)dim * ldx1 + row * ldx2];
          float v = x * scales[split_base + s] + offsets[split_base + s];
          int iv = (int)roundf(x);
          if (iv > 127) iv = 127;
          else if (iv < -128) iv = -128;
          int8_t x_i8 = (int8_t)iv;
          const int8_t* tbl = splitvals + vals_per_split * (split_base + s);
          int8_t threshold = tbl[code];
          int bit = (x_i8 > threshold) ? 1 : 0;
          code = (code << 1) | bit;
        }
        // Write the code to the output buffer
        out[c * m + row] = (uint8_t)code;
      }
    }
  }
}
// ~~ Decoder ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
