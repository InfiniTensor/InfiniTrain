add_executable(conv2d_test conv2d_test.cc)
target_link_libraries(conv2d_test PRIVATE infini_train gtest gtest_main)
add_test(NAME conv2d_test COMMAND conv2d_test)