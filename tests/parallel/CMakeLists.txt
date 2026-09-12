# ==========================================================================
# Parallel topology tests
# ==========================================================================

infini_train_add_test(test_parallel_topology
  SOURCES test_global_env.cc
  TEST_MAIN_SOURCE test_global_env_main.cc
  LABELS cpu
)
