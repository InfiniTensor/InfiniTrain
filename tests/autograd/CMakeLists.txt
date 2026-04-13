# ============================================================================
# Autograd tests
# ============================================================================

set(AUTOGRAD_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB AUTOGRAD_SOURCES ${AUTOGRAD_TEST_DIR}/test_autograd*.cc)

infini_train_add_test_suite(test_autograd
  SOURCES ${AUTOGRAD_SOURCES}
)
