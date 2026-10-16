# ==========================================================================
# Parallel tests
# ==========================================================================

infini_train_add_test(test_parallel_topology
  SOURCES test_global_env.cc
  TEST_MAIN_SOURCE test_global_env_main.cc
  LABELS cpu
)

if(USE_CUDA AND USE_NCCL)
  infini_train_add_test(test_moe_ep_cuda
    SOURCES cuda_only/test_moe_ep_correctness.cc
    TEST_MAIN_SOURCE cuda_only/test_moe_ep_correctness_main.cc
    LABELS cuda distributed moe
    TEST_TIMEOUT 120
    COMPILE_DEFINITIONS INFINI_TRAIN_TEST_EP=2
    BINARY_REGISTRATION
    RUN_SERIAL
  )
endif()
