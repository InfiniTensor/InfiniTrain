# ==========================================================================
# Optimizer tests
# ==========================================================================

set(OPTIMIZER_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(OPTIMIZER_TEST_SOURCES
  ${OPTIMIZER_TEST_DIR}/test_optimizer_creation.cc
  ${OPTIMIZER_TEST_DIR}/test_optimizer_step.cc
)

infini_train_add_test_suite(test_optimizer
  SOURCES ${OPTIMIZER_TEST_SOURCES}
)
