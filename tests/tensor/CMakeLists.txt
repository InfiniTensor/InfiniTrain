# Tensor tests (single binary)

set(TENSOR_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TENSOR_TEST_SOURCES
  ${TENSOR_TEST_DIR}/test_tensor.cc
)

add_executable(test_tensor ${TENSOR_TEST_SOURCES})
target_compile_options(test_tensor PRIVATE -Wno-error)
link_infini_train_exe(test_tensor)
target_link_libraries(test_tensor PRIVATE GTest::gtest GTest::gtest_main)
target_include_directories(test_tensor PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
  ${glog_SOURCE_DIR}/src
)

include(GoogleTest)
gtest_discover_tests(test_tensor
  EXTRA_ARGS --gtest_output=xml:%T.xml
  TEST_FILTER "-TensorDistributedTest.*:*CUDA*:TensorOpTest.*"
  PROPERTIES LABELS "cpu"
)
gtest_discover_tests(test_tensor
  EXTRA_ARGS --gtest_output=xml:%T.xml
  TEST_FILTER "TensorOpTest.*:*CUDA*"
  PROPERTIES LABELS "cuda"
)
gtest_discover_tests(test_tensor
  EXTRA_ARGS --gtest_output=xml:%T.xml
  TEST_FILTER "TensorDistributedTest.*"
  PROPERTIES LABELS "distributed"
)
