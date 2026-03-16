# Optimizer tests

# Helper function to add optimizer test executable
function(add_optimizer_test name sources)
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

# Optimizer creation tests
add_optimizer_test(test_optimizer_creation
  test_optimizer_creation.cc
)
add_test(NAME optimizer_create COMMAND test_optimizer_creation)
set_tests_properties(optimizer_create PROPERTIES LABELS "cpu")

# Optimizer step tests
add_optimizer_test(test_optimizer_step
  test_optimizer_step.cc
)
add_test(NAME optimizer_step COMMAND test_optimizer_step)
set_tests_properties(optimizer_step PROPERTIES LABELS "cpu")

# Optimizer CUDA tests
add_optimizer_test(test_optimizer_cuda
  test_optimizer_cuda.cc
)
add_test(NAME optimizer_cuda COMMAND test_optimizer_cuda)
set_tests_properties(optimizer_cuda PROPERTIES LABELS "cuda")

# Optimizer distributed tests
add_optimizer_test(test_optimizer_distributed
  test_optimizer_distributed.cc
)
add_test(NAME optimizer_distributed COMMAND test_optimizer_distributed)
set_tests_properties(optimizer_distributed PROPERTIES LABELS "cuda;distributed")

# Legacy combined tests (for backward compatibility)
add_executable(test_optimizer_legacy
  test_optimizer.cc
)
target_link_libraries(test_optimizer_legacy
  PRIVATE
    GTest::gtest
    GTest::gtest_main
)
target_include_directories(test_optimizer_legacy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
target_link_libraries(test_optimizer_legacy PRIVATE
  "-Wl,--whole-archive"
  infini_train
  infini_train_cpu_kernels
  "-Wl,--no-whole-archive"
)
add_test(NAME optimizer_create_legacy COMMAND test_optimizer_legacy --gtest_filter=OptimizerCreationTest.*)
set_tests_properties(optimizer_create_legacy PROPERTIES LABELS "cpu")
add_test(NAME optimizer_zero_grad_legacy COMMAND test_optimizer_legacy --gtest_filter=OptimizerGradTest.*)
set_tests_properties(optimizer_zero_grad_legacy PROPERTIES LABELS "cpu")
add_test(NAME optimizer_cuda_legacy COMMAND test_optimizer_legacy --gtest_filter=OptimizerCudaTest.*)
set_tests_properties(optimizer_cuda_legacy PROPERTIES LABELS "cuda")
add_test(NAME optimizer_distributed_legacy COMMAND test_optimizer_legacy --gtest_filter=OptimizerDistributedTest.*)
set_tests_properties(optimizer_distributed_legacy PROPERTIES LABELS "cuda;distributed")

# Aggregate target for convenient builds
add_custom_target(test_optimizer
  DEPENDS
    test_optimizer_creation
    test_optimizer_step
    test_optimizer_cuda
    test_optimizer_distributed
    test_optimizer_legacy
)
