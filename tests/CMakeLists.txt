set(TEST_TARGET_NAME amm_test)
set(TEST_SOURCES
  test_ndarray.c)

add_executable(${TEST_TARGET_NAME} ${TEST_SOURCES})
target_include_directories(${TEST_TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)
set_target_properties(${TEST_TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
