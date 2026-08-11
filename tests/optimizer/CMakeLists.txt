# ==========================================================================
# Optimizer tests
# ==========================================================================

file(GLOB OPTIMIZER_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cc)

infini_train_add_test_suite(test_optimizer
  SOURCES ${OPTIMIZER_SOURCES}
)

add_test(
  NAME OptimizerParameterNamesTest.DistributedOptimizerPropagatesNamesToShardOptimizer
  COMMAND ${CMAKE_COMMAND} -E env
    "PROC_WORLD_SIZE=2"
    $<TARGET_FILE:test_optimizer_cuda>
    --gtest_filter=CUDA/OptimizerParameterNamesTest.DistributedOptimizerPropagatesNamesToShardOptimizer/*
)
set_tests_properties(
  OptimizerParameterNamesTest.DistributedOptimizerPropagatesNamesToShardOptimizer
  PROPERTIES
    LABELS "cuda;distributed"
    TIMEOUT 30
)
