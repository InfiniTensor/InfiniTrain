# Tensor tests

add_executable(test_tensor
  test_tensor.cc
)
target_compile_options(test_tensor PRIVATE -Wno-error)
target_link_libraries(test_tensor
  PRIVATE
    GTest::gtest
    GTest::gtest_main
)
target_include_directories(test_tensor PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
  ${glog_SOURCE_DIR}/src
)

target_link_libraries(test_tensor PRIVATE
  "-Wl,--whole-archive"
  infini_train
  infini_train_cpu_kernels
  "-Wl,--no-whole-archive"
)

add_test(NAME tensor_cpu COMMAND test_tensor)
set_tests_properties(tensor_cpu PROPERTIES LABELS "cpu")

add_test(NAME tensor_cuda COMMAND test_tensor --gtest_filter=TensorTest.*CUDA)
set_tests_properties(tensor_cuda PROPERTIES LABELS "cuda")

add_test(NAME tensor_distributed COMMAND test_tensor --gtest_filter=TensorTest.*Distributed)
set_tests_properties(tensor_distributed PROPERTIES LABELS "cuda;distributed")
