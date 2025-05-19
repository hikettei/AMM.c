cmake_minimum_required(VERSION 3.14)

project("AMM.c" C)
set(CMAKE_CXX_STANDARD 17)

set(AMM_C_TARGET amm_c)

# === Kernel Options ===============================================================================
option(AMM_C_NAIVE  "amm.c: Use NAIVE" ON)
option(AMM_C_AVX2   "amm.c: Use AVX2"   OFF)
option(AMM_C_AVX512 "amm.c: Use AVX512" OFF)

option(AMM_C_ARM64  "amm.c: Use Arm64" OFF)

option(AMM_C_METAL  "amm.c: Use METAL" OFF)
option(AMM_C_CUDA   "amm.c: Use CUDA"  OFF)
# ==================================================================================================
# == Build (Common Parts) ==========================================================================
set(AMM_C_SOURCES
    src/encoding.c
    src/maddness.c)
add_executable(${AMM_C_TARGET} ${AMM_C_SOURCES})
# ==================================================================================================
# == Build (Kernel Parts) ==========================================================================
# Note(hikettei): If you want to add a new backend named <BACKEND> you may have to do the following:
# 1. Add a new option AMM_C_<BACKEND> in the kernel options above.
# 2. Create a new file named ./src/kernels/<backend>/CMakeLists.txt, and complete your implementation.
#    - Here, add_library(${AMM_C_KERNEL_<BACKEND>_TARGET} ...) must be specified.
# 3. Add amm_register_backend(<backend>)

# e.g.: the directory (related to hardware-specific stuff parts) should look like:
#  └── src
#    ├── kernels
#    │   ├── arm64
#    │   │   ├── CMakeLists.txt
#    │   │   └── ...
#    │   └── naive
#    │       ├── CMakeLists.txt
#    │       └── ...
#        ...

function(amm_register_backend backend)
    string(TOUPPER "${backend}" backend_upper)
    string(TOUPPER "AMM_C_KERNEL_${backend}_TARGET" backend_tgt_nmsp)
    string(TOLOWER "AMM_C_KERNEL_${backend}" backend_tgt_lw_nmsp)
    set(option_name "AMM_C_${backend_upper}")
    set(${backend_tgt_nmsp} ${backend_tgt_lw_nmsp})
    if(${option_name})
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/${backend}")
        # common header
        target_include_directories(${backend_tgt_lw_nmsp} PRIVATE ./include/kernels)
        target_link_libraries(${AMM_C_TARGET} ${backend_tgt_lw_nmsp})
    endif()
endfunction()

amm_register_backend(naive)
amm_register_backend(avx2)
amm_register_backend(avx512)
amm_register_backend(arm64)
amm_register_backend(metal)
amm_register_backend(cuda)
