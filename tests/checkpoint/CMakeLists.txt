# ==========================================================================
# Checkpoint tests
# ==========================================================================

file(GLOB CHECKPOINT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cc)

infini_train_add_test_suite(test_checkpoint
  SOURCES ${CHECKPOINT_SOURCES}
)
