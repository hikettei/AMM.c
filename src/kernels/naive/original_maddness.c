#include "comm_original_maddness.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <inttypes.h>

/*
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

  assert(m % block_nrows == 0); // m must be a multiple of 32
  
  int total_nsplits = ncodebooks * nsplits_per_codebook;
  int maxdim = splitdims[0], mindim = splitdims[0];

  for (int i = 1; i < total_nsplits; i++) {
    if (splitdims[i] > (uint32_t)maxdim) maxdim = splitdims[i];
    if (splitdims[i] < (uint32_t)mindim) mindim = splitdims[i];
  }
  assert(mindim >= 0 && maxdim < n);
  int64_t nblocks = m / block_nrows;
  
  for (int c = 0; c < ncodebooks; c++) {
    int split_base = c * nsplits_per_codebook;
    for (int b = 0; b < nblocks; b++) {
      for (int i = 0; i < block_nrows; i++) {
        int64_t row = (int64_t)b * block_nrows + i;
        int code = 0;
        for (int s = 0; s < nsplits_per_codebook; s++) {
          uint32_t dim = splitdims[split_base + s];
          float x = X[(int64_t)dim * ldx + row];
          float v = x * scales[split_base + s] + offsets[split_base + s];
          // Quantization with saturation (round → clamp to [-128,127])
          // TODO: Use an arbitary quantization algorithm
          // Varying from 2 bits to 8 bits
          // Using different algorithm inspired from CV Quantization
          // ↑ Trying Channelwise Quantization
          int iv = (int)roundf(v);
          if      (iv > 127) iv = 127;
          else if (iv < -128) iv = -128;
          int8_t x_i8 = (int8_t)iv;
          const int8_t* tbl = splitvals + vals_per_split * (split_base + s);
          int8_t threshold = tbl[code];
          int bit = (x_i8 > threshold) ? 1 : 0;
          code = (code << 1) | bit;
        }
        out[c * m + row] = (uint8_t)code;
      }
    }
  }
}
*/

/*
// Scan and Aggregation
template <int NBytes, int UpcastEvery = 16, int _OutTileSz = 1, bool Force16BitOutput = false>
void scan_m_f32(
    const uint8_t* codes,
    int64_t        nblocks,
    const uint8_t* luts,
    uint8_t*       dists_out)
{
    static_assert(NBytes > 0,                    "Code length <= 0 is not valid");
    static_assert(UpcastEvery % 2 == 0,          "UpcastEvery must be even");
    static_assert(UpcastEvery >= 2,              "UpcastEvery must be >= 2");
    static_assert(UpcastEvery <= 256,            "UpcastEvery must be <= 256");
    static_assert((UpcastEvery & (UpcastEvery-1)) == 0,
                  "UpcastEvery must be a power of 2");

    constexpr int ncodebooks = 2 * NBytes;
    constexpr int ncols      = NBytes;
    constexpr int actually_upcast_every = std::min(UpcastEvery, ncodebooks);
    constexpr int colgroup_sz          = actually_upcast_every / 2;
    static_assert((colgroup_sz & (colgroup_sz-1)) == 0,
                  "colgroup_sz must be a power of 2");
    static_assert(ncols % colgroup_sz == 0,
                  "ncols must be divisible by colgroup_sz");
    constexpr int ncolgroups = ncols / colgroup_sz;
    constexpr bool use_uint8_output = (ncolgroups == 1 && !Force16BitOutput);
    constexpr int OutTileSz = (_OutTileSz > 0 ? _OutTileSz : 1);

    // 各タイル mm あたりの LUT ブロック長 (バイト)
    const int lut_stride = ncodebooks * 16;
    // 出力ストライド (バイト)
    const int64_t out_stride = use_uint8_output
        ? (nblocks * 32)
        : (nblocks * 32 * 2);

    // タイルごとの出力ポインタを準備
    const uint8_t* lut_base = luts;
    uint8_t* out_ptrs[OutTileSz];
    for (int mm = 0; mm < OutTileSz; ++mm) {
        out_ptrs[mm] = dists_out + mm * out_stride;
    }

    // LUT ポインタをタイル×コードブックで平滑化しておく
    // lut_ptrs[mm][k] は、タイル mm のコードブック k (0～ncodebooks-1) に対応する
    // 16 バイトのルックアップテーブル先頭を指す
    const uint8_t* lut_ptrs[OutTileSz][ncodebooks];
    for (int mm = 0; mm < OutTileSz; ++mm) {
        const uint8_t* p = lut_base + mm * lut_stride;
        // 各コードブックごとに 16 バイト分ずつ並んでいる
        for (int k = 0; k < ncodebooks; ++k) {
            lut_ptrs[mm][k] = p + k * 16;
        }
    }

    // codes ポインタは連続して nblocks × ncols × 32 バイト分データを読む
    const uint8_t* code_ptr = codes;

    // メイン：各ブロックごとに
    for (int64_t blk = 0; blk < nblocks; ++blk) {
        // 16 ビット出力時の合計用バッファ
        int16_t totals_lo[OutTileSz][32] = {0};
        int16_t totals_hi[OutTileSz][32] = {0};

        // 各列グループごとに集計
        for (int g = 0; g < ncolgroups; ++g) {
            // 各タイル mm、各レーン i について「このグループでの合計」をためる
            int sum[OutTileSz][32] = {0};

            // グループ内の各サブ列 (gg) を走査
            for (int gg = 0; gg < colgroup_sz; ++gg) {
                int j = g * colgroup_sz + gg;
                // 32 行分のコードバイトを読む
                uint8_t local_codes[32];
                for (int i = 0; i < 32; ++i) {
                    local_codes[i] = *code_ptr++;
                }

                // 各タイル mm ごとに距離 avg を計算して sum[mm][i] に足す
                for (int mm = 0; mm < OutTileSz; ++mm) {
                    const uint8_t* lut_low  = lut_ptrs[mm][2*j + 0];
                    const uint8_t* lut_high = lut_ptrs[mm][2*j + 1];
                    for (int i = 0; i < 32; ++i) {
                        uint8_t code = local_codes[i];
                        uint8_t lo = code & 0x0F;
                        uint8_t hi = (code >> 4) & 0x0F;
                        uint8_t dist_lo  = lut_low [lo];
                        uint8_t dist_hi  = lut_high[hi];
                        // u8 平均 (round up on ties): (a + b + 1) / 2
                        uint8_t avg = uint8_t((int)dist_lo + (int)dist_hi + 1) >> 1;
                        sum[mm][i] += avg;
                    }
                }
            } // end for gg

            // グループサイズで割って最終 avg を得る
            for (int mm = 0; mm < OutTileSz; ++mm) {
                uint8_t group_avg[32];
                for (int i = 0; i < 32; ++i) {
                    // 四捨五入： (sum + colgroup_sz/2) / colgroup_sz
                    int v = sum[mm][i] + (colgroup_sz/2);
                    group_avg[i] = uint8_t(v / colgroup_sz);
                }

                if (use_uint8_output) {
                    // そのまま 32 バイト／ブロック出力
                    for (int i = 0; i < 32; ++i) {
                        out_ptrs[mm][i] = group_avg[i];
                    }
                    out_ptrs[mm] += 32;
                } else {
                    // 16 ビット出力用に合計値をさらに累積
                    for (int i = 0; i < 32; ++i) {
                        if (i < 16) totals_lo[mm][i] += group_avg[i];
                        else        totals_hi[mm][i-16] += group_avg[i];
                    }
                }
            }
        } // end for g

        // 16 ビット出力の場合、まとめて書き出し
        if (!use_uint8_output) {
            for (int mm = 0; mm < OutTileSz; ++mm) {
                // lower 16 lanes
                for (int i = 0; i < 16; ++i) {
                    uint16_t v = uint16_t(totals_lo[mm][i]);
                    *out_ptrs[mm]++ = uint8_t(v & 0xFF);
                    *out_ptrs[mm]++ = uint8_t(v >> 8);
                }
                // upper 16 lanes
                for (int i = 0; i < 16; ++i) {
                    uint16_t v = uint16_t(totals_hi[mm][i]);
                    *out_ptrs[mm]++ = uint8_t(v & 0xFF);
                    *out_ptrs[mm]++ = uint8_t(v >> 8);
                }
            }
        }
    } // end for blk
}
*/
