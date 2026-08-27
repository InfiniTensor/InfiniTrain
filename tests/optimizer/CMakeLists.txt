# ==========================================================================
# Optimizer tests
# ==========================================================================

file(GLOB OPTIMIZER_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cc)

infini_train_add_test_suite(test_optimizer
  SOURCES ${OPTIMIZER_SOURCES}
)
