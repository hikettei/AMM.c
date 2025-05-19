cmake_minimum_required(VERSION 3.14)


project("AMM.cpp" C CXX)
set(CMAKE_CXX_STANDARD 17)

# === Build Target ==========================================================
option(VISION_LANGUAGE_CPP_USE_CUDA "Use CUDA(through cuew loader)" OFF)

option(AMM_CPP_AVX2   "amm.cpp: Use AVX2"   OFF)
option(AMM_CPP_AVX512 "amm.cpp: Use AVX512" OFF)

option(AMM_CPP_ARM64  "amm.cpp: Use Arm64" OFF)

option(AMM_CPP_METAL  "amm.cpp: Use METAL" OFF)
option(AMM_CPP_CUDA   "amm.cpp: Use CUDA"  OFF)
# ===========================================================================
# == Build (Kernel Parts) ===================================================

# ===========================================================================
# == Build (Common Parts) ===================================================

# ===========================================================================
