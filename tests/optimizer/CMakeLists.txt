# ==========================================================================
# Optimizer tests
# ==========================================================================

file(GLOB OPTIMIZER_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cc)

infini_train_add_test_suite(test_optimizer
  SOURCES ${OPTIMIZER_SOURCES}
)

add_test(
  NAME OptimizerParameterNamesTest.DistributedOptimizerPropagatesNamesToShardOptimizer
  COMMAND ${CMAKE_COMMAND} -E env INFINI_RUN_ID=optimizer_ctest torchrun --standalone --no-python --nproc_per_node=2
    $<TARGET_FILE:test_optimizer_cuda>
    --gtest_filter=CUDA/OptimizerParameterNamesTest.DistributedOptimizerPropagatesNamesToShardOptimizer/*
)
set_tests_properties(
  OptimizerParameterNamesTest.DistributedOptimizerPropagatesNamesToShardOptimizer
  PROPERTIES
    LABELS "cuda;distributed"
    TIMEOUT 30
)

add_test(
  NAME OptimizerParameterNamesTest.DistributedOptimizerClipGradNormUsesGlobalShardNorm
  COMMAND ${CMAKE_COMMAND} -E env INFINI_RUN_ID=optimizer_ctest torchrun --standalone --no-python --nproc_per_node=2
    $<TARGET_FILE:test_optimizer_cuda>
    --gtest_filter=CUDA/OptimizerParameterNamesTest.DistributedOptimizerClipGradNormUsesGlobalShardNorm/*
)
set_tests_properties(
  OptimizerParameterNamesTest.DistributedOptimizerClipGradNormUsesGlobalShardNorm
  PROPERTIES
    LABELS "cuda;distributed"
    TIMEOUT 30
)

add_test(
  NAME OptimizerParameterNamesTest.DistributedOptimizerClipGradNormUsesZero2LocalShard
  COMMAND ${CMAKE_COMMAND} -E env INFINI_RUN_ID=optimizer_zero2_ctest torchrun --standalone --no-python --nproc_per_node=2
    $<TARGET_FILE:test_optimizer_cuda>
    --gtest_filter=CUDA/OptimizerParameterNamesTest.DistributedOptimizerClipGradNormUsesZero2LocalShard/*
)
set_tests_properties(
  OptimizerParameterNamesTest.DistributedOptimizerClipGradNormUsesZero2LocalShard
  PROPERTIES
    LABELS "cuda;distributed"
    TIMEOUT 30
)

add_test(
  NAME OptimizerParameterNamesTest.DistributedOptimizerClipGradNormHandlesEmptyLocalShard
  COMMAND ${CMAKE_COMMAND} -E env INFINI_RUN_ID=optimizer_empty_ctest torchrun --standalone --no-python --nproc_per_node=2
    $<TARGET_FILE:test_optimizer_cuda>
    --gtest_filter=CUDA/OptimizerParameterNamesTest.DistributedOptimizerClipGradNormHandlesEmptyLocalShard/*
)
set_tests_properties(
  OptimizerParameterNamesTest.DistributedOptimizerClipGradNormHandlesEmptyLocalShard
  PROPERTIES
    LABELS "cuda;distributed"
    TIMEOUT 30
)
