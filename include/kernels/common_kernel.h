
/*
  include/kernels/common_encoder.h
  Common parts for the encoder.
  (TODO: Naming convention following SLEEF's style)
  ammcpp_encode_<algorithm>_<dtype>_<lut_dtype>_<quantizer>
  Dtype:
    float -> f32
    int8 -> i8
     ...

  Note:
  Encoder = Quantize + Create LUT
  Decoder = Aggregate from LUT + Dequantize
*/
#include <stdint.h>

#define define_amm_encoder(algorithm, in_dtype_prefix, in_dtype, lut_dtype_prefix, lut_dtype, quantizer) \
     void amm_encode_##algorithm##_##in_dtype_prefix##_##lut_dtype_prefix##_##quantizer( \
      const in_dtype *X, int m, int n, int ldx, /* X is a matrix of size [m, n] with the leading dimension ldx */ \
      int ncodebooks, /* Training Configuration: Number of codebooks (C) */ \
      const uint32_t *splitdims, const int8_t *splitvals, const in_dtype *scales, const in_dtype *offsets, /* Quantization Parameters */ \
      lut_dtype* out /* An allocated buffer for storing the quantized LUT sized [C, n] */); \

// TODO
#define define_amm_decoder(algorithm, lut_dtype_prefix, lut_dtype, quantizer) \
      void amm_scan_##lut_dtype_prefix##_##quantizer( \
        const in_dtype *X, int m, int n, int ldx, /* X is a matrix of size [m, n] with the leading dimension ldx */ \
        int ncodebooks, /* Training Configuration: Number of codebooks (C) */ \
        const uint32_t *splitdims, const int8_t *splitvals, const in_dtype *scales, const in_dtype *offsets, /* Quantization Parameters */ \
        lut_dtype* out /* An allocated buffer for storing the quantized LUT sized [C, n] */); \

// Encoders
define_amm_encoder(m, f32, float, i8, uint8_t, clamp)
// TODO: More Follows ...

// Scanners
// void scan_m_f32(const uint8_t* encoded_mat, int C, int M, const uint8_t* luts, uint8_t* out_mat);
