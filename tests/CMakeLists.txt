function(define_test tgt_name tgt_file)
  add_executable(${tgt_name} ${tgt_file})
  target_include_directories(${tgt_name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(${tgt_name} PRIVATE ${AMM_C_TARGET})
  set_target_properties(${tgt_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endfunction()

define_test(amm_test_ndarray test_ndarray.c)
define_test(amm_test_amm1 test_amm1.c)
