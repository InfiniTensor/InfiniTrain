# ==========================================================================
# Hook tests
# ==========================================================================

file(GLOB HOOK_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cc)

infini_train_add_test_suite(test_hook
  SOURCES ${HOOK_SOURCES}
)
