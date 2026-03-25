# Tensor tests

# test_tensor_create
add_executable(test_tensor_create
  test_tensor_create.cc
)
target_compile_options(test_tensor_create PRIVATE -Wno-error)
target_link_libraries(test_tensor_create
  PRIVATE
    GTest::gtest
    GTest::gtest_main
)
target_include_directories(test_tensor_create PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
  ${glog_SOURCE_DIR}/src
)
target_link_libraries(test_tensor_create PRIVATE
  "-Wl,--whole-archive"
  infini_train
  infini_train_cpu_kernels
  "-Wl,--no-whole-archive"
)
add_test(NAME tensor_create COMMAND test_tensor_create)
set_tests_properties(tensor_create PROPERTIES LABELS "cpu;cuda")

# test_tensor_copy
add_executable(test_tensor_copy
  test_tensor_copy.cc
)
target_compile_options(test_tensor_copy PRIVATE -Wno-error)
target_link_libraries(test_tensor_copy
  PRIVATE
    GTest::gtest
    GTest::gtest_main
)
target_include_directories(test_tensor_copy PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
  ${glog_SOURCE_DIR}/src
)
target_link_libraries(test_tensor_copy PRIVATE
  "-Wl,--whole-archive"
  infini_train
  infini_train_cpu_kernels
  "-Wl,--no-whole-archive"
)
add_test(NAME tensor_copy COMMAND test_tensor_copy)
set_tests_properties(tensor_copy PROPERTIES LABELS "cpu;cuda")

# test_tensor_delete
add_executable(test_tensor_delete
  test_tensor_delete.cc
)
target_compile_options(test_tensor_delete PRIVATE -Wno-error)
target_link_libraries(test_tensor_delete
  PRIVATE
    GTest::gtest
    GTest::gtest_main
)
target_include_directories(test_tensor_delete PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
  ${glog_SOURCE_DIR}/src
)
target_link_libraries(test_tensor_delete PRIVATE
  "-Wl,--whole-archive"
  infini_train
  infini_train_cpu_kernels
  "-Wl,--no-whole-archive"
)
add_test(NAME tensor_delete COMMAND test_tensor_delete)
set_tests_properties(tensor_delete PROPERTIES LABELS "cpu")

# test_tensor_op (keep original for backward compatibility)
add_executable(test_tensor_op
  test_tensor.cc
)
target_compile_options(test_tensor_op PRIVATE -Wno-error)
target_link_libraries(test_tensor_op
  PRIVATE
    GTest::gtest
    GTest::gtest_main
)
target_include_directories(test_tensor_op PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
  ${glog_SOURCE_DIR}/src
)
target_link_libraries(test_tensor_op PRIVATE
  "-Wl,--whole-archive"
  infini_train
  infini_train_cpu_kernels
  "-Wl,--no-whole-archive"
)
add_test(NAME tensor_ops COMMAND test_tensor_op --gtest_filter=TensorOpTest.*)
set_tests_properties(tensor_ops PROPERTIES LABELS "cpu;cuda")

# test_tensor_distributed (keep original for backward compatibility)
add_executable(test_tensor_dist
  test_tensor.cc
)
target_compile_options(test_tensor_dist PRIVATE -Wno-error)
target_link_libraries(test_tensor_dist
  PRIVATE
    GTest::gtest
    GTest::gtest_main
)
target_include_directories(test_tensor_dist PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
  ${glog_SOURCE_DIR}/src
)
target_link_libraries(test_tensor_dist PRIVATE
  "-Wl,--whole-archive"
  infini_train
  infini_train_cpu_kernels
  "-Wl,--no-whole-archive"
)
add_test(NAME tensor_distributed COMMAND test_tensor_dist --gtest_filter=TensorDistributedTest.*)
set_tests_properties(tensor_distributed PROPERTIES LABELS "cuda;distributed")

# Convenience aggregate target so `cmake --build ... --target test_tensor` works
add_custom_target(test_tensor
  DEPENDS
    test_tensor_create
    test_tensor_copy
    test_tensor_delete
    test_tensor_op
    test_tensor_dist
)
