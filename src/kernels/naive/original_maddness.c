#include "comm_original_maddness.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <inttypes.h>

#ifndef MIN
#define MIN(a,b) (( (a) < (b) ? (a) : (b) ))
#endif


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
void mithral_scan(
    const uint8_t *codes,         // [nblocks * NBytes * 32]
    int64_t        nblocks,
    const uint8_t *luts,          // [OutTileSz==1 ? (2*NBytes*16) : ignored tiling]
    uint8_t       *dists_out,     // 出力バッファ
    int            NBytes,
    int            UpcastEvery,
    int            Force16BitOutput  // 0 or 1
) {
    // --- パラメータチェック ---
    assert(NBytes > 0);
    assert((UpcastEvery % 2) == 0);
    assert(UpcastEvery >= 2 && UpcastEvery <= 256);

    int ncodebooks = 2 * NBytes;             // low/hi LUT がそれぞれ NBytes 個ずつ
    int ncols      = NBytes;
    int actually_upcast_every = MIN(UpcastEvery, ncodebooks);
    int colgroup_sz = actually_upcast_every / 2;
    assert(colgroup_sz > 0 && (colgroup_sz & (colgroup_sz - 1)) == 0);
    assert(ncols % colgroup_sz == 0);
    int ncolgroups = ncols / colgroup_sz;

    int use_uint8_output = (ncolgroups == 1 && !Force16BitOutput);

    // LUT を 16 バイトずつ指すポインタ配列を準備
    // (テンプレート版ではタイル数分あったが、ここでは tile=1 扱い)
    const uint8_t *lut_ptrs[/*max*/ 512];  // 十分大きめに確保
    {
        const uint8_t *p = luts;
        for (int k = 0; k < ncodebooks; ++k) {
            lut_ptrs[k] = p;
            p += 16;
        }
    }

    const uint8_t *code_ptr = codes;

    // --- 各ブロックごとに ---
    for (int64_t blk = 0; blk < nblocks; ++blk) {
        // 16bit 出力時の累積用バッファ
        int16_t totals_lo[16] = {0};
        int16_t totals_hi[16] = {0};

        // 列グループごとに処理
        for (int g = 0; g < ncolgroups; ++g) {
            // このグループ内の各行 32 レーン分を累積する一時バッファ
            int sum32[32] = {0};

            // グループ内のサブ列を走査
            for (int gg = 0; gg < colgroup_sz; ++gg) {
                int j = g * colgroup_sz + gg;
                // codes から 32 バイトを読み出し
                uint8_t local_codes[32];
                for (int i = 0; i < 32; ++i) {
                    local_codes[i] = *code_ptr++;
                }
                // low/hi LUT ポインタ
                const uint8_t *lut_low  = lut_ptrs[2*j + 0];
                const uint8_t *lut_high = lut_ptrs[2*j + 1];

                // 32 レーン分、距離を足し込む
                for (int i = 0; i < 32; ++i) {
                    uint8_t code = local_codes[i];
                    uint8_t lo   = code & 0x0F;
                    uint8_t hi   = (code >> 4) & 0x0F;
                    uint8_t d_lo = lut_low [lo];
                    uint8_t d_hi = lut_high[hi];
                    // u8 平均 (切り上げ方向に丸め)
                    sum32[i] += ((int)d_lo + (int)d_hi + 1) >> 1;
                }
            }

            // グループ平均を出して出力 or 累積
            {
                uint8_t avg32[32];
                for (int i = 0; i < 32; ++i) {
                    // 四捨五入平均
                    avg32[i] = (uint8_t)((sum32[i] + colgroup_sz/2) / colgroup_sz);
                }

                if (use_uint8_output) {
                    // そのまま 32 バイト書き出し
                    for (int i = 0; i < 32; ++i) {
                        *dists_out++ = avg32[i];
                    }
                } else {
                    // 16 ビット出力用に下位16 と上位16 を累積
                    for (int i = 0; i < 16; ++i) {
                        totals_lo[i] += avg32[i];
                    }
                    for (int i = 16; i < 32; ++i) {
                        totals_hi[i-16] += avg32[i];
                    }
                }
            }
        }

        // 16 ビット出力時はまとめて書き出し
        if (!use_uint8_output) {
            for (int i = 0; i < 16; ++i) {
                uint16_t v_lo = (uint16_t)totals_lo[i];
                uint16_t v_hi = (uint16_t)totals_hi[i];
                // リトルエンディアンで 2 バイトずつ
                *dists_out++ = (uint8_t)(v_lo & 0xFF);
                *dists_out++ = (uint8_t)(v_lo >> 8);
            }
            for (int i = 0; i < 16; ++i) {
                uint16_t v_lo = (uint16_t)totals_hi[i];
                *dists_out++ = (uint8_t)(v_lo & 0xFF);
                *dists_out++ = (uint8_t)(v_lo >> 8);
            }
        }
    }
}
