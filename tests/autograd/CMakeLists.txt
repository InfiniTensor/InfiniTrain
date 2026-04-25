# ============================================================================
# Autograd tests
# ============================================================================

file(GLOB AUTOGRAD_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cc)

infini_train_add_test_suite(test_autograd
  SOURCES ${AUTOGRAD_SOURCES}
)
