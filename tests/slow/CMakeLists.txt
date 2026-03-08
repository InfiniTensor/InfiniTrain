# Slow label smoke tests

add_executable(test_slow
  test_slow.cc
)
target_link_libraries(test_slow
  PRIVATE
    GTest::gtest
    GTest::gtest_main
)
target_include_directories(test_slow PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)

target_link_libraries(test_slow PRIVATE
  "-Wl,--whole-archive"
  infini_train
  infini_train_cpu_kernels
  "-Wl,--no-whole-archive"
)

add_test(NAME slow_cpu COMMAND test_slow --gtest_filter=SlowTest.Cpu)
set_tests_properties(slow_cpu PROPERTIES LABELS "slow;cpu")

add_test(NAME slow_cuda COMMAND test_slow --gtest_filter=SlowTest.Cuda)
set_tests_properties(slow_cuda PROPERTIES LABELS "slow;cuda")

add_test(NAME slow_distributed COMMAND test_slow --gtest_filter=SlowTest.Distributed)
set_tests_properties(slow_distributed PROPERTIES LABELS "slow;cuda;distributed")
