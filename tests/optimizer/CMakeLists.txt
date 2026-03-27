# ==========================================================================
# Optimizer tests (single binary)
# ==========================================================================

# 注意：test_optimizer.cc 为 legacy 组合测试，内容与拆分文件重复，故未纳入此二进制

set(OPTIMIZER_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(OPTIMIZER_TEST_SOURCES
  ${OPTIMIZER_TEST_DIR}/test_optimizer_creation.cc
  ${OPTIMIZER_TEST_DIR}/test_optimizer_step.cc
  ${OPTIMIZER_TEST_DIR}/test_optimizer_cuda.cc
  ${OPTIMIZER_TEST_DIR}/test_optimizer_distributed.cc
)

add_executable(test_optimizer ${OPTIMIZER_TEST_SOURCES})
target_compile_options(test_optimizer PRIVATE -Wno-error)
link_infini_train_exe(test_optimizer)
target_link_libraries(test_optimizer PRIVATE GTest::gtest GTest::gtest_main)
target_include_directories(test_optimizer PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
  ${glog_SOURCE_DIR}/src
)

include(GoogleTest)
gtest_discover_tests(test_optimizer
  EXTRA_ARGS --gtest_output=xml:%T.xml
  TEST_FILTER "-OptimizerCUDATest.*:OptimizerDistributedTest.*"
  PROPERTIES LABELS "cpu"
)
gtest_discover_tests(test_optimizer
  EXTRA_ARGS --gtest_output=xml:%T.xml
  TEST_FILTER "OptimizerCUDATest.*"
  PROPERTIES LABELS "cuda"
)
gtest_discover_tests(test_optimizer
  EXTRA_ARGS --gtest_output=xml:%T.xml
  TEST_FILTER "OptimizerDistributedTest.*"
  PROPERTIES LABELS "distributed"
)
