# Hook tests

add_executable(test_hook
  test_hook.cc
)
target_link_libraries(test_hook
  PRIVATE
    GTest::gtest
    GTest::gtest_main
)
target_include_directories(test_hook PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)

target_link_libraries(test_hook PRIVATE
  "-Wl,--whole-archive"
  infini_train
  infini_train_cpu_kernels
  "-Wl,--no-whole-archive"
)

add_test(NAME hook_cpu COMMAND test_hook)
set_tests_properties(hook_cpu PROPERTIES LABELS "cpu")

add_test(NAME hook_cuda COMMAND test_hook --gtest_filter=HookTest.*CUDA)
set_tests_properties(hook_cuda PROPERTIES LABELS "cuda")

add_test(NAME hook_distributed COMMAND test_hook --gtest_filter=HookTest.*Distributed)
set_tests_properties(hook_distributed PROPERTIES LABELS "cuda;distributed")

add_executable(test_precision_check
  test_precision_check.cc
)
target_link_libraries(test_precision_check
  PRIVATE
    GTest::gtest
    GTest::gtest_main
)
target_include_directories(test_precision_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)

target_link_libraries(test_precision_check PRIVATE
  "-Wl,--whole-archive"
  infini_train
  infini_train_cpu_kernels
  "-Wl,--no-whole-archive"
)

add_test(NAME precision_check_cpu COMMAND test_precision_check)
set_tests_properties(precision_check_cpu PROPERTIES LABELS "cpu")
