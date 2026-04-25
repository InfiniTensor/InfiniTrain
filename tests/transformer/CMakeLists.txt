# ==========================================================================
# Transformer architecture tests
# ==========================================================================

file(GLOB TRANSFORMER_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cc)

infini_train_add_test_suite(test_transformer
  SOURCES ${TRANSFORMER_SOURCES}
)
