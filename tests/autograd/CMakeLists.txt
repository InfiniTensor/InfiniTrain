# ============================================================================
# Autograd tests
# ============================================================================
# 重构版本：使用单一 test binary 聚合测试源码，减少 target 膨胀
# ============================================================================

# -----------------------------------------------------------------------------
# Autograd tests (single binary)
# -----------------------------------------------------------------------------
set(AUTOGRAD_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(AUTOGRAD_TEST_SOURCES
  ${AUTOGRAD_TEST_DIR}/test_autograd_elementwise_forward.cc
  ${AUTOGRAD_TEST_DIR}/test_autograd_elementwise_backward.cc
  ${AUTOGRAD_TEST_DIR}/test_autograd_matmul_forward.cc
  ${AUTOGRAD_TEST_DIR}/test_autograd_matmul_backward.cc
  ${AUTOGRAD_TEST_DIR}/test_autograd_reduction_forward.cc
  ${AUTOGRAD_TEST_DIR}/test_autograd_reduction_backward.cc
  ${AUTOGRAD_TEST_DIR}/test_autograd_linear_forward.cc
  ${AUTOGRAD_TEST_DIR}/test_autograd_linear_backward.cc
  ${AUTOGRAD_TEST_DIR}/test_autograd_softmax_forward.cc
  ${AUTOGRAD_TEST_DIR}/test_autograd_softmax_backward.cc
  ${AUTOGRAD_TEST_DIR}/test_autograd_transform_forward.cc
  ${AUTOGRAD_TEST_DIR}/test_autograd_transform_backward.cc
  ${AUTOGRAD_TEST_DIR}/test_autograd_normalization_forward.cc
  ${AUTOGRAD_TEST_DIR}/test_autograd_normalization_backward.cc
  ${AUTOGRAD_TEST_DIR}/test_autograd.cc
)

add_executable(test_autograd ${AUTOGRAD_TEST_SOURCES})
target_compile_options(test_autograd PRIVATE -Wno-error)
link_infini_train_exe(test_autograd)
target_link_libraries(test_autograd PRIVATE GTest::gtest GTest::gtest_main)
target_include_directories(test_autograd
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${glog_SOURCE_DIR}/src
)

include(GoogleTest)
gtest_discover_tests(test_autograd
  EXTRA_ARGS --gtest_output=xml:%T.xml
  TEST_FILTER "-AutogradCudaTest.*:AutogradDistributedTest.*"
  PROPERTIES LABELS "cpu"
)
gtest_discover_tests(test_autograd
  EXTRA_ARGS --gtest_output=xml:%T.xml
  TEST_FILTER "AutogradCudaTest.*"
  PROPERTIES LABELS "cuda"
)
gtest_discover_tests(test_autograd
  EXTRA_ARGS --gtest_output=xml:%T.xml
  TEST_FILTER "AutogradDistributedTest.*"
  PROPERTIES LABELS "distributed"
)
