# ==========================================================================
# Hook tests
# ==========================================================================

set(HOOK_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(HOOK_TEST_SOURCES
  ${HOOK_TEST_DIR}/test_hook.cc
  ${HOOK_TEST_DIR}/test_precision_check.cc
)

infini_train_add_test_suite(test_hook
  SOURCES ${HOOK_TEST_SOURCES}
)
