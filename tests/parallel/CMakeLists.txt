# Topology tests run in a standalone process because GlobalEnv is initialized once.
add_executable(test_parallel_topology test_global_env.cc)
target_link_libraries(test_parallel_topology PRIVATE GTest::gtest)
link_infini_train_exe(test_parallel_topology)
gtest_discover_tests(test_parallel_topology
  PROPERTIES LABELS cpu TIMEOUT 10
)
