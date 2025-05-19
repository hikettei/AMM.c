typedef struct {
  int N, M, K; // A[N M] @ B[M K]  
  int LDX; // Column Major or Row Major.
  int C; // Number of Codebooks
  int nsplits; // Number of splits per codebook

  void* quantized_lut; // TODO: Quantizes into int8_t ~ binary/ternary?
} OriginalMaddnessGemm;
