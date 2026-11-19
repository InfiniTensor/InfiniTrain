add_executable(test_thread_link test_thread_link.cc)
target_link_libraries(test_thread_link PRIVATE InfiniTrain::infini_train_executable)
add_test(NAME Link.ThreadDependency COMMAND test_thread_link)
set_tests_properties(Link.ThreadDependency PROPERTIES LABELS "cpu;link" TIMEOUT 10)
