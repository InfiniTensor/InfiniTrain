# ==========================================================================
# Hook tests (single binary)
# ==========================================================================

set(HOOK_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(HOOK_TEST_SOURCES
  ${HOOK_TEST_DIR}/test_hook.cc
  ${HOOK_TEST_DIR}/test_precision_check.cc
)

add_executable(test_hook ${HOOK_TEST_SOURCES})
target_compile_options(test_hook PRIVATE -Wno-error)
link_infini_train_exe(test_hook)
target_link_libraries(test_hook PRIVATE GTest::gtest GTest::gtest_main)
target_include_directories(test_hook PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
  ${glog_SOURCE_DIR}/src
)

include(GoogleTest)
gtest_discover_tests(test_hook
  EXTRA_ARGS --gtest_output=xml:%T.xml
  TEST_FILTER "-*CUDA*:*Distributed*"
  PROPERTIES LABELS "cpu"
)
gtest_discover_tests(test_hook
  EXTRA_ARGS --gtest_output=xml:%T.xml
  TEST_FILTER "*CUDA*"
  PROPERTIES LABELS "cuda"
)
gtest_discover_tests(test_hook
  EXTRA_ARGS --gtest_output=xml:%T.xml
  TEST_FILTER "*Distributed*"
  PROPERTIES LABELS "distributed"
)
