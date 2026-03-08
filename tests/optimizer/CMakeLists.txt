# Optimizer tests

add_executable(test_optimizer
  test_optimizer.cc
)
target_link_libraries(test_optimizer
  PRIVATE
    GTest::gtest
    GTest::gtest_main
)
target_include_directories(test_optimizer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)

target_link_libraries(test_optimizer PRIVATE
  "-Wl,--whole-archive"
  infini_train
  infini_train_cpu_kernels
  "-Wl,--no-whole-archive"
)

add_test(NAME optimizer_cpu COMMAND test_optimizer)
set_tests_properties(optimizer_cpu PROPERTIES LABELS "cpu")

add_test(NAME optimizer_cuda COMMAND test_optimizer --gtest_filter=OptimizerTest.*CUDA)
set_tests_properties(optimizer_cuda PROPERTIES LABELS "cuda")

add_test(NAME optimizer_distributed COMMAND test_optimizer --gtest_filter=OptimizerTest.*Distributed)
set_tests_properties(optimizer_distributed PROPERTIES LABELS "cuda;distributed")
