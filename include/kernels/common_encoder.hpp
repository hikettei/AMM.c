
/*
maddness_encode_dtype creates a lut.
maddness_scan_dtype looks up the lut and encodes the data.
*/

// Naming Convention: encode_<algorithm>_<dtype>

void encode_m_f32(const float *X, int m, int n, int ldx,
                  // X is a matrix of sized [m, n] with the leading dimension ldx
                  int ncodebooks, // Training Configuration: Number of codebooks (C)
                  const uint32_t *splitdims, const int8_t *splitvals, const float *scales, const float *offsets, // Quantization Parameters
                  uint8_t* out); // An allocated buffer for storing the quantized LUT sized [C, n]

void scan_m_f32(const uint8_t* encoded_mat, int C, int M, const uint8_t* luts, uint8_t* out_mat);
