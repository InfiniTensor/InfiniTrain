# ==========================================================================
# Slow tests (single binary)
# ==========================================================================

set(SLOW_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(SLOW_TEST_SOURCES
  ${SLOW_TEST_DIR}/test_slow.cc
)

add_executable(test_slow ${SLOW_TEST_SOURCES})
target_compile_options(test_slow PRIVATE -Wno-error)
link_infini_train_exe(test_slow)
target_link_libraries(test_slow PRIVATE GTest::gtest GTest::gtest_main)
target_include_directories(test_slow PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
  ${glog_SOURCE_DIR}/src
)

include(GoogleTest)
gtest_discover_tests(test_slow
  EXTRA_ARGS --gtest_output=xml:%T.xml
  TEST_FILTER "SlowTest.Cpu"
  PROPERTIES LABELS "slow;cpu"
)
gtest_discover_tests(test_slow
  EXTRA_ARGS --gtest_output=xml:%T.xml
  TEST_FILTER "SlowTest.Cuda"
  PROPERTIES LABELS "slow;cuda"
)
gtest_discover_tests(test_slow
  EXTRA_ARGS --gtest_output=xml:%T.xml
  TEST_FILTER "SlowTest.Distributed"
  PROPERTIES LABELS "slow;distributed"
)
