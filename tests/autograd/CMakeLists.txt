# Autograd operators test

add_executable(test_autograd
  test_autograd.cc
)
target_link_libraries(test_autograd
  PRIVATE
    GTest::gtest
    GTest::gtest_main
)
target_include_directories(test_autograd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)

target_link_libraries(test_autograd PRIVATE
  "-Wl,--whole-archive"
  infini_train
  infini_train_cpu_kernels
  "-Wl,--no-whole-archive"
)

add_test(NAME autograd_cpu COMMAND test_autograd)
set_tests_properties(autograd_cpu PROPERTIES LABELS "cpu")

add_test(NAME autograd_cuda COMMAND test_autograd --gtest_filter=AutogradTest.*CUDA)
set_tests_properties(autograd_cuda PROPERTIES LABELS "cuda")

add_test(NAME autograd_distributed COMMAND test_autograd --gtest_filter=AutogradTest.*Distributed)
set_tests_properties(autograd_distributed PROPERTIES LABELS "cuda;distributed")
