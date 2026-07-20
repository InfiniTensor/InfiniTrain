# ==========================================================================
# Parallel layout tests
# ==========================================================================

file(GLOB PARALLEL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cc)
list(FILTER PARALLEL_SOURCES EXCLUDE REGEX "test_global_env(_etp)?\\.cc$")

infini_train_add_test(test_parallel
  SOURCES ${PARALLEL_SOURCES}
  LABELS cpu
)

# This target owns its main function so it can initialize GlobalEnv with a
# non-trivial TP/EP topology in a fresh process.
add_executable(test_parallel_env test_global_env.cc)
target_link_libraries(test_parallel_env PRIVATE GTest::gtest)
link_infini_train_exe(test_parallel_env)
gtest_discover_tests(test_parallel_env
  PROPERTIES LABELS cpu TIMEOUT 10
)

# Validate a topology where dense TP and expert TP (ETP) differ.
add_executable(test_parallel_etp_env test_global_env_etp.cc)
target_link_libraries(test_parallel_etp_env PRIVATE GTest::gtest)
link_infini_train_exe(test_parallel_etp_env)
gtest_discover_tests(test_parallel_etp_env
  PROPERTIES LABELS cpu TIMEOUT 10
)
