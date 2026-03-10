# Autograd tests

# Helper function to add autograd test executable
function(add_autograd_test name sources)
  add_executable(${name} ${sources})
  target_link_libraries(${name}
    PRIVATE
      GTest::gtest
      GTest::gtest_main
  )
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
  target_link_libraries(${name} PRIVATE
    "-Wl,--whole-archive"
    infini_train
    infini_train_cpu_kernels
    "-Wl,--no-whole-archive"
  )
endfunction()

# Elementwise forward tests
add_autograd_test(test_autograd_elementwise_forward
  test_autograd_elementwise_forward.cc
)
add_test(NAME autograd_elementwise_forward COMMAND test_autograd_elementwise_forward)
set_tests_properties(autograd_elementwise_forward PROPERTIES LABELS "cpu")

# Elementwise backward tests
add_autograd_test(test_autograd_elementwise_backward
  test_autograd_elementwise_backward.cc
)
add_test(NAME autograd_elementwise_backward COMMAND test_autograd_elementwise_backward)
set_tests_properties(autograd_elementwise_backward PROPERTIES LABELS "cpu")

# Matmul forward tests
add_autograd_test(test_autograd_matmul_forward
  test_autograd_matmul_forward.cc
)
add_test(NAME autograd_matmul_forward COMMAND test_autograd_matmul_forward)
set_tests_properties(autograd_matmul_forward PROPERTIES LABELS "cpu")

# Matmul backward tests
add_autograd_test(test_autograd_matmul_backward
  test_autograd_matmul_backward.cc
)
add_test(NAME autograd_matmul_backward COMMAND test_autograd_matmul_backward)
set_tests_properties(autograd_matmul_backward PROPERTIES LABELS "cpu")

# Reduction forward tests
add_autograd_test(test_autograd_reduction_forward
  test_autograd_reduction_forward.cc
)
add_test(NAME autograd_reduction_forward COMMAND test_autograd_reduction_forward)
set_tests_properties(autograd_reduction_forward PROPERTIES LABELS "cpu")

# Reduction backward tests
add_autograd_test(test_autograd_reduction_backward
  test_autograd_reduction_backward.cc
)
add_test(NAME autograd_reduction_backward COMMAND test_autograd_reduction_backward)
set_tests_properties(autograd_reduction_backward PROPERTIES LABELS "cpu")

# Linear forward tests
add_autograd_test(test_autograd_linear_forward
  test_autograd_linear_forward.cc
)
add_test(NAME autograd_linear_forward COMMAND test_autograd_linear_forward)
set_tests_properties(autograd_linear_forward PROPERTIES LABELS "cpu")

# Linear backward tests
add_autograd_test(test_autograd_linear_backward
  test_autograd_linear_backward.cc
)
add_test(NAME autograd_linear_backward COMMAND test_autograd_linear_backward)
set_tests_properties(autograd_linear_backward PROPERTIES LABELS "cpu")

# Softmax forward tests
add_autograd_test(test_autograd_softmax_forward
  test_autograd_softmax_forward.cc
)
add_test(NAME autograd_softmax_forward COMMAND test_autograd_softmax_forward)
set_tests_properties(autograd_softmax_forward PROPERTIES LABELS "cpu")

# Softmax backward tests
add_autograd_test(test_autograd_softmax_backward
  test_autograd_softmax_backward.cc
)
add_test(NAME autograd_softmax_backward COMMAND test_autograd_softmax_backward)
set_tests_properties(autograd_softmax_backward PROPERTIES LABELS "cpu")

# Transform forward tests
add_autograd_test(test_autograd_transform_forward
  test_autograd_transform_forward.cc
)
add_test(NAME autograd_transform_forward COMMAND test_autograd_transform_forward)
set_tests_properties(autograd_transform_forward PROPERTIES LABELS "cpu")

# Transform backward tests
add_autograd_test(test_autograd_transform_backward
  test_autograd_transform_backward.cc
)
add_test(NAME autograd_transform_backward COMMAND test_autograd_transform_backward)
set_tests_properties(autograd_transform_backward PROPERTIES LABELS "cpu")

# Normalization forward tests
add_autograd_test(test_autograd_normalization_forward
  test_autograd_normalization_forward.cc
)
add_test(NAME autograd_normalization_forward COMMAND test_autograd_normalization_forward)
set_tests_properties(autograd_normalization_forward PROPERTIES LABELS "cpu")

# Normalization backward tests
add_autograd_test(test_autograd_normalization_backward
  test_autograd_normalization_backward.cc
)
add_test(NAME autograd_normalization_backward COMMAND test_autograd_normalization_backward)
set_tests_properties(autograd_normalization_backward PROPERTIES LABELS "cpu")

# Legacy combined tests (for backward compatibility)
add_executable(test_autograd_legacy
  test_autograd.cc
)
target_link_libraries(test_autograd_legacy
  PRIVATE
    GTest::gtest
    GTest::gtest_main
)
target_include_directories(test_autograd_legacy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(test_autograd_legacy PRIVATE
  "-Wl,--whole-archive"
  infini_train
  infini_train_cpu_kernels
  "-Wl,--no-whole-archive"
)
add_test(NAME autograd_forward_legacy COMMAND test_autograd_legacy --gtest_filter=AutogradForwardTest.*)
set_tests_properties(autograd_forward_legacy PROPERTIES LABELS "cpu")
add_test(NAME autograd_backward_legacy COMMAND test_autograd_legacy --gtest_filter=AutogradBackwardTest.*)
set_tests_properties(autograd_backward_legacy PROPERTIES LABELS "cpu")
add_test(NAME autograd_cuda_legacy COMMAND test_autograd_legacy --gtest_filter=AutogradCudaTest.*)
set_tests_properties(autograd_cuda_legacy PROPERTIES LABELS "cuda")
add_test(NAME autograd_distributed_legacy COMMAND test_autograd_legacy --gtest_filter=AutogradDistributedTest.*)
set_tests_properties(autograd_distributed_legacy PROPERTIES LABELS "cuda;distributed")
